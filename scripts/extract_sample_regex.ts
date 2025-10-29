import { parse, stringify } from "@std/csv";

import { ROOT_DIR } from "../utils/directory.ts";
import { join } from "@std/path/join";

type CSVRow = {
    repo: string;
    issue_number: number;
    issue_closed: boolean;
    closing_prs_count: number;
}

type QualifiedIssue = CSVRow & {
    qualification_reason: "linked_closing_prs" | "prs_in_discussion";
}

type PRDetails = {
    hash_refs: number[];
    pr_urls: string[];
    linked_closing_prs?: number[];
    all_pr_numbers: number[];
    all_pr_urls: string[];
}

type IssueWithPRDetails = QualifiedIssue & PRDetails;

type RepoSummary = {
    repo: string;
    total_issues: number;
    issues_with_linked_prs: number;
    issues_with_discussion_prs: number;
    issues: IssueWithPRDetails[];
}

type JSONReport = {
    summary: {
        total_repos: number;
        total_qualified_issues: number;
        issues_with_linked_prs: number;
        issues_with_discussion_prs: number;
        generated_at: string;
    };
    repositories: RepoSummary[];
}

async function importCSV(filePath: string): Promise<CSVRow[]> {
    const csvPath = join(ROOT_DIR, "output", filePath);
    const csvContent = await Deno.readTextFile(csvPath);
    const rows = parse(csvContent, { skipFirstRow: true });
    return rows.map(row => ({
        repo: row.repo,
        issue_number: Number(row.issue_number),
        issue_closed: row.issue_closed.toLowerCase() === 'true',
        closing_prs_count: Number(row.closing_prs_count)
    }));
}

async function importClosingPRsCSV(): Promise<Map<string, number[]>> {
    const csvPath = join(ROOT_DIR, "output", "issues_closing_prs.csv");
    const csvContent = await Deno.readTextFile(csvPath);
    const rows = parse(csvContent, { skipFirstRow: true });
    
    const closingPRsMap = new Map<string, number[]>();
    
    for (const row of rows) {
        const repo = row.repo;
        const issueNumber = row.issue_number;
        const closingPRsStr = row.closing_pr_numbers;
        
        // Parse the closing_pr_numbers format: [] or [317] or [221 223]
        const prNumbers: number[] = [];
        if (closingPRsStr && closingPRsStr !== '[]') {
            const match = closingPRsStr.match(/\[(.*?)\]/);
            if (match && match[1].trim()) {
                const numbers = match[1].trim().split(/\s+/);
                for (const numStr of numbers) {
                    const num = Number(numStr);
                    if (!Number.isNaN(num)) {
                        prNumbers.push(num);
                    }
                }
            }
        }
        
        closingPRsMap.set(`${repo}#${issueNumber}`, prNumbers);
    }
    
    return closingPRsMap;
}

function extractClosedIssues(records: CSVRow[]) {
    return records.filter(record => record.issue_closed);
}

function extractIssuesWithClosedPRs(records: CSVRow[]) {
    return records.filter(record => record.closing_prs_count > 0);
}

function extractIssuesWithoutClosedPrs(records: CSVRow[]) {
    return records.filter(record => record.closing_prs_count === 0);
}

async function extractPRDetails(record: CSVRow): Promise<PRDetails> {
    const repoDir = record.repo.replace("/", "+");
    const filePath = join(ROOT_DIR, "output", "issues_text", repoDir, `issue_${record.issue_number}.txt`);
    
    try {
        const text = await Deno.readTextFile(filePath);
        
        // Extract hash references like #123 but exclude:
        // - Comment headers like "#1:" at start of line or after "---"
        // - Issue titles like "#742:" at start of text
        // - Sequential comment numbers
        // Look for #number that appears in actual text context (not headers)
        const hashNumberRe = /(?<!^|\n|---\s*)(?<=\s)#(\d+)(?!\s*:)/gm;
        const hashRefs = new Set<number>();
        for (const match of text.matchAll(hashNumberRe)) {
            const num = Number(match[1]);
            if (!Number.isNaN(num)) {
                hashRefs.add(num);
            }
        }
        
        // Extract PR URLs and extract PR numbers from them
        const prUrlRe = /https?:\/\/github\.com\/[\w.-]+\/[\w.-]+\/pulls?\/(\d+)/gi;
        const prUrls = new Set<string>();
        const prNumbersFromUrls = new Set<number>();
        for (const match of text.matchAll(prUrlRe)) {
            const cleaned = match[0].replace(/[),.;:]+$/g, ""); // Remove trailing punctuation
            prUrls.add(cleaned);
            const prNum = Number(match[1]);
            if (!Number.isNaN(prNum)) {
                prNumbersFromUrls.add(prNum);
            }
        }
        
        // Combine all PR numbers from hash refs and URLs
        const allPRNumbers = new Set([...hashRefs, ...prNumbersFromUrls]);
        
        // Generate GitHub PR URLs for all PR numbers using the current repository
        const allPRUrls = Array.from(allPRNumbers).map(num => 
            `https://github.com/${record.repo}/pull/${num}`
        ).sort();
        
        return {
            hash_refs: Array.from(hashRefs).sort((a, b) => a - b),
            pr_urls: Array.from(prUrls).sort(),
            all_pr_numbers: Array.from(allPRNumbers).sort((a, b) => a - b),
            all_pr_urls: allPRUrls
        };
    } catch {
        return {
            hash_refs: [],
            pr_urls: [],
            all_pr_numbers: [],
            all_pr_urls: []
        };
    }
}

async function checkIssueForPRPatterns(record: CSVRow): Promise<boolean> {
    const repoDir = record.repo.replace("/", "+");
    const filePath = join(ROOT_DIR, "output", "issues_text", repoDir, `issue_${record.issue_number}.txt`);
    
    try {
        const text = await Deno.readTextFile(filePath);
        
        // Look for #number that is preceded by space and NOT followed by colon
        // AND NOT at the start of a line or after "---" (to exclude comment headers)
        const hashNumberRe = /(?<!^|\n|---\s*)(?<=\s)#(\d+)(?!\s*:)/gm;
        const prUrlRe = /https?:\/\/github\.com\/[\w.-]+\/[\w.-]+\/pulls?\/\d+/gi;
        
        const hasHashRefs = hashNumberRe.test(text);
        const hasPrUrls = prUrlRe.test(text);
        
        return hasHashRefs || hasPrUrls;
    } catch {
        return false;
    }
}

async function extractIssuesWithPRsInDiscussion(records: CSVRow[]): Promise<CSVRow[]> {
    const qualified: CSVRow[] = [];
    for (const record of records) {
        if (await checkIssueForPRPatterns(record)) {
            qualified.push(record);
        }
    }
    return qualified;
}

function qualifyIssues(issuesWithClosingPRs: CSVRow[], issuesWithDiscussionPRs: CSVRow[]): QualifiedIssue[] {
    const qualified: QualifiedIssue[] = [];
    
    for (const issue of issuesWithClosingPRs) {
        qualified.push({
            ...issue,
            qualification_reason: "linked_closing_prs"
        });
    }
    
    const existingKeys = new Set(qualified.map(q => `${q.repo}#${q.issue_number}`));
    for (const issue of issuesWithDiscussionPRs) {
        const key = `${issue.repo}#${issue.issue_number}`;
        if (!existingKeys.has(key)) {
            qualified.push({
                ...issue,
                qualification_reason: "prs_in_discussion"
            });
        }
    }
    
    return qualified;
}

async function enrichIssuesWithPRDetails(qualifiedIssues: QualifiedIssue[]): Promise<IssueWithPRDetails[]> {
    const enriched: IssueWithPRDetails[] = [];
    
    // Load the closing PRs data
    const closingPRsMap = await importClosingPRsCSV();
    
    for (const issue of qualifiedIssues) {
        const prDetails = await extractPRDetails(issue);
        
        // Add linked closing PRs if this issue has them
        const issueKey = `${issue.repo}#${issue.issue_number}`;
        const linkedClosingPRs = closingPRsMap.get(issueKey) || [];
        
        // Combine all PR numbers: from discussion + linked closing PRs
        const allPRNumbers = new Set([...prDetails.all_pr_numbers, ...linkedClosingPRs]);
        const allPRUrls = Array.from(allPRNumbers).map(num => 
            `https://github.com/${issue.repo}/pull/${num}`
        ).sort();
        
        enriched.push({
            ...issue,
            ...prDetails,
            linked_closing_prs: linkedClosingPRs.length > 0 ? linkedClosingPRs : undefined,
            all_pr_numbers: Array.from(allPRNumbers).sort((a, b) => a - b),
            all_pr_urls: allPRUrls
        });
    }
    
    return enriched;
}

async function generateJSONReport(enrichedIssues: IssueWithPRDetails[], destDirName: string): Promise<void> {
    // Group issues by repository
    const repoMap = new Map<string, IssueWithPRDetails[]>();
    for (const issue of enrichedIssues) {
        if (!repoMap.has(issue.repo)) {
            repoMap.set(issue.repo, []);
        }
        repoMap.get(issue.repo)!.push(issue);
    }
    
    // Create repository summaries
    const repositories: RepoSummary[] = [];
    for (const [repo, issues] of repoMap.entries()) {
        const linkedPRCount = issues.filter(i => i.qualification_reason === "linked_closing_prs").length;
        const discussionPRCount = issues.filter(i => i.qualification_reason === "prs_in_discussion").length;
        
        repositories.push({
            repo,
            total_issues: issues.length,
            issues_with_linked_prs: linkedPRCount,
            issues_with_discussion_prs: discussionPRCount,
            issues: issues.sort((a, b) => a.issue_number - b.issue_number)
        });
    }
    
    // Sort repositories by name
    repositories.sort((a, b) => a.repo.localeCompare(b.repo));
    
    // Create final report
    const report: JSONReport = {
        summary: {
            total_repos: repositories.length,
            total_qualified_issues: enrichedIssues.length,
            issues_with_linked_prs: enrichedIssues.filter(i => i.qualification_reason === "linked_closing_prs").length,
            issues_with_discussion_prs: enrichedIssues.filter(i => i.qualification_reason === "prs_in_discussion").length,
            generated_at: new Date().toISOString()
        },
        repositories
    };
    
    // Write JSON file
    const destRoot = join(ROOT_DIR, "output", destDirName);
    const jsonPath = join(destRoot, "issues_pr_details.json");
    await Deno.writeTextFile(jsonPath, JSON.stringify(report, null, 2));
    
    console.log(`Generated JSON report: ${jsonPath}`);
}

async function copyIssueText(record: QualifiedIssue, destDirName: string): Promise<boolean> {
    const repoDir = record.repo.replace("/", "+");
    const src = join(ROOT_DIR, "output", "issues_text", repoDir, `issue_${record.issue_number}.txt`);
    const destDir = join(ROOT_DIR, "output", destDirName, repoDir);
    const dest = join(destDir, `issue_${record.issue_number}.txt`);
    
    try {
        await Deno.mkdir(destDir, { recursive: true });
        await Deno.copyFile(src, dest);
        return true;
    } catch {
        return false;
    }
}

async function copyAllIssueTexts(records: QualifiedIssue[], destDirName: string): Promise<number> {
    let copied = 0;
    for (const record of records) {
        if (await copyIssueText(record, destDirName)) {
            copied++;
        }
    }
    return copied;
}

async function copyFilteredCSV(qualifiedIssues: QualifiedIssue[], destDirName: string): Promise<void> {
    const destRoot = join(ROOT_DIR, "output", destDirName);
    await Deno.mkdir(destRoot, { recursive: true });
    
    const originalCSVPath = join(ROOT_DIR, "output", "issues_report.csv");
    const csvContent = await Deno.readTextFile(originalCSVPath);
    const originalRows = parse(csvContent, { skipFirstRow: true });
    
    const qualificationMap = new Map<string, string>();
    for (const issue of qualifiedIssues) {
        qualificationMap.set(`${issue.repo}#${issue.issue_number}`, issue.qualification_reason);
    }

    const enhancedRows = [];
    for (const row of originalRows) {
        const repo = row.repo;
        const issueNum = row.issue_number;
        const key = `${repo}#${issueNum}`;
        const reason = qualificationMap.get(key);
        if (reason) {
            enhancedRows.push({
                repo: row.repo,
                issue_number: row.issue_number,
                issue_url: row.issue_url,
                issue_title: row.issue_title,
                number_of_comments: row.number_of_comments,
                total_reactions_in_issue: row.total_reactions_in_issue,
                total_reactions_in_comments: row.total_reactions_in_comments,
                closed_by_pull_requests: row.closed_by_pull_requests,
                qualification_reason: reason
            });
        }
    }
    
    const csvOutput = stringify(enhancedRows, {
        columns: [
            "repo", "issue_number", "issue_url", "issue_title", 
            "number_of_comments", "total_reactions_in_issue",
            "total_reactions_in_comments", "closed_by_pull_requests",
            "qualification_reason"
        ]
    });
    
    const destCSVPath = join(destRoot, "issues_report.csv");
    await Deno.writeTextFile(destCSVPath, csvOutput);
}

async function run() {
    const issues = await importCSV("issues_status.csv");
    const closedIssues = extractClosedIssues(issues);
    const issuesWithClosedPRs = extractIssuesWithClosedPRs(closedIssues);
    const remainingClosedIssues = extractIssuesWithoutClosedPrs(closedIssues);

    console.log("Scanning for PR patterns in discussions...");
    const issuesWithDiscussionPRs = await extractIssuesWithPRsInDiscussion(remainingClosedIssues);
    console.log(`Found ${issuesWithDiscussionPRs.length} issues with PR patterns in discussion`);
    
    const qualifiedIssues = qualifyIssues(issuesWithClosedPRs, issuesWithDiscussionPRs);
    console.log(`Total qualified issues: ${qualifiedIssues.length}`);
    
    if (qualifiedIssues.length > 0) {
        const enrichedIssues = await enrichIssuesWithPRDetails(qualifiedIssues);
        
        const copiedFiles = await copyAllIssueTexts(qualifiedIssues, "issues_text_closed");
        await copyFilteredCSV(qualifiedIssues, "issues_text_closed");
        await generateJSONReport(enrichedIssues, "issues_text_closed");
        
        console.log({
            closed_issues_total: closedIssues.length,
            issues_with_closed_prs: issuesWithClosedPRs.length,
            issues_with_discussion_prs: issuesWithDiscussionPRs.length,
            qualified_issues_total: qualifiedIssues.length,
            copied_files: copiedFiles,
            dest_dir: join(ROOT_DIR, "output", "issues_text_closed"),
        });
    } else {
        console.log("No qualified issues found.");
    }
}

if (import.meta.main) {
    await run();
}