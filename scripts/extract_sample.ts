import { join } from "@std/path";
import { parse, stringify } from "@std/csv";
import { ROOT_DIR } from "../utils/directory.ts";

const SAMPLE_SIZE_N = 2; 
const OUTPUT_DIR_NAME = "sample_output_N2_2"; 

interface IssueRecord {
	repo: string;
	issue_number: number;
	issue_url: string;
	issue_title: string;
	number_of_comments: number;
	total_reactions_in_issue: number;
	total_reactions_in_comments: number;
	closed_by_pull_requests: number;
}

interface RepoSample {
	repo: string;
	issues: IssueRecord[];
}

async function parseCSVFile(): Promise<Map<string, IssueRecord[]>> {
	const csvPath = join(ROOT_DIR, "output", "issues_report.csv");
	const csvContent = await Deno.readTextFile(csvPath);
	
	const records = parse(csvContent, { 
		skipFirstRow: true,
		columns: [
			"repo",
			"issue_number", 
			"issue_url",
			"issue_title",
			"number_of_comments",
			"total_reactions_in_issue", 
			"total_reactions_in_comments",
			"closed_by_pull_requests"
		]
	});
	
	const repoIssues = new Map<string, IssueRecord[]>();
	
	for (const record of records) {
		const issueRecord: IssueRecord = {
			repo: record.repo as string,
			issue_number: parseInt(record.issue_number as string),
			issue_url: record.issue_url as string,
			issue_title: record.issue_title as string,
			number_of_comments: parseInt(record.number_of_comments as string),
			total_reactions_in_issue: parseInt(record.total_reactions_in_issue as string),
			total_reactions_in_comments: parseInt(record.total_reactions_in_comments as string),
			closed_by_pull_requests: parseInt(record.closed_by_pull_requests as string)
		};
		
		if (!repoIssues.has(issueRecord.repo)) {
			repoIssues.set(issueRecord.repo, []);
		}
		repoIssues.get(issueRecord.repo)!.push(issueRecord);
	}
	
	return repoIssues;
}


function randomSample<T>(array: T[], n: number): T[] {
	if (array.length <= n) {
		return [...array];
	}
	
	const shuffled = [...array];
	for (let i = shuffled.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
	}
	
	return shuffled.slice(0, n);
}

async function copyIssueTextFiles(repoSamples: RepoSample[], outputDir: string): Promise<void> {
	const textOutputDir = join(outputDir, "issues_text");
	
	for (const sample of repoSamples) {
		const repoDir = sample.repo.replace("/", "+");
		const sourceDir = join(ROOT_DIR, "output", "issues_text", repoDir);
		const targetDir = join(textOutputDir, repoDir);
		
		await Deno.mkdir(targetDir, { recursive: true });
		
		for (const issue of sample.issues) {
			const sourceFile = join(sourceDir, `issue_${issue.issue_number}.txt`);
			const targetFile = join(targetDir, `issue_${issue.issue_number}.txt`);
			
			try {
				await Deno.copyFile(sourceFile, targetFile);
			} catch (error) {
				console.warn(`Could not copy ${sourceFile}: ${error instanceof Error ? error.message : String(error)}`);
			}
		}
		
		console.log(`Copied ${sample.issues.length} text files for ${sample.repo}`);
	}
}

async function generateSampleCSV(repoSamples: RepoSample[], outputDir: string): Promise<void> {
	const csvPath = join(outputDir, "issues_report.csv");

	const allSampledIssues = repoSamples.flatMap(sample => sample.issues);
	const csvData = allSampledIssues.map(issue => ({
		repo: issue.repo,
		issue_number: issue.issue_number,
		issue_url: issue.issue_url,
		issue_title: issue.issue_title,
		number_of_comments: issue.number_of_comments,
		total_reactions_in_issue: issue.total_reactions_in_issue,
		total_reactions_in_comments: issue.total_reactions_in_comments,
		closed_by_pull_requests: issue.closed_by_pull_requests
	}));
	
	const csvContent = stringify(csvData, {
		columns: [
			"repo",
			"issue_number", 
			"issue_url",
			"issue_title",
			"number_of_comments",
			"total_reactions_in_issue", 
			"total_reactions_in_comments",
			"closed_by_pull_requests"
		]
	});
	
	await Deno.writeTextFile(csvPath, csvContent);
	console.log(`Generated sample CSV with ${allSampledIssues.length} issues`);
}

async function generateSummaryReport(repoSamples: RepoSample[], outputDir: string): Promise<void> {
	const summaryPath = join(outputDir, "sampling_summary.txt");
	
	let summary = `Sampling Summary\n`;
	summary += `================\n\n`;
	summary += `Sample size per repository: ${SAMPLE_SIZE_N}\n`;
	summary += `Total repositories: ${repoSamples.length}\n`;
	summary += `Total sampled issues: ${repoSamples.reduce((sum, sample) => sum + sample.issues.length, 0)}\n\n`;
	
	summary += `Repository breakdown:\n`;
	summary += `---------------------\n`;
	
	for (const sample of repoSamples) {
		summary += `${sample.repo}: ${sample.issues.length} issues\n`;
	}
	
	summary += `\nGenerated on: ${new Date().toISOString()}\n`;
	
	await Deno.writeTextFile(summaryPath, summary);
	console.log("Generated sampling summary");
}


async function extractSample(): Promise<void> {
	console.log(`Starting sample extraction with N=${SAMPLE_SIZE_N}`);
	
	console.log("Parsing issues report CSV...");
	const repoIssues = await parseCSVFile();
	console.log(`Found ${repoIssues.size} repositories`);
	
	const repoSamples: RepoSample[] = [];
	for (const [repo, issues] of repoIssues.entries()) {
		const sampledIssues = randomSample(issues, SAMPLE_SIZE_N);
		repoSamples.push({
			repo,
			issues: sampledIssues
		});
		console.log(`Sampled ${sampledIssues.length}/${issues.length} issues from ${repo}`);
	}
	
	const outputDir = join(ROOT_DIR, "output", OUTPUT_DIR_NAME);
	await Deno.mkdir(outputDir, { recursive: true });
	console.log(`Created output directory: ${outputDir}`);
	
	console.log("Copying issue text files...");
	await copyIssueTextFiles(repoSamples, outputDir);
	
	console.log("Generating sample CSV report...");
	await generateSampleCSV(repoSamples, outputDir);
	
	console.log("Generating summary report...");
	await generateSummaryReport(repoSamples, outputDir);
	
	const totalSampled = repoSamples.reduce((sum, sample) => sum + sample.issues.length, 0);
	console.log(`\nSample extraction completed successfully!`);
	console.log(`Output directory: ${outputDir}`);
	console.log(`Total issues sampled: ${totalSampled}`);
	console.log(`Repositories processed: ${repoSamples.length}`);
}

if (import.meta.main) {
	await extractSample();
}
