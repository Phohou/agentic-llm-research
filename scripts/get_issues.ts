import { Octokit } from "https://esm.sh/octokit";
import "jsr:@std/dotenv/load";
import { join } from "@std/path";
import { ROOT_DIR } from "../utils/directory.ts";
import { createDirNameFromRepo } from "../utils/repo.ts";

const octokit = new Octokit({ auth: Deno.env.get("GITHUB_TOKEN_1") });

const repoConfig = {
	"run-llama/llama_index" : {
		labels: null,
	},
	"microsoft/autogen" : {
		labels: null,
	},
	"crewAIInc/crewAI" : {
		labels: null,
	},
	"microsoft/semantic-kernel" : {
		labels : null,
	},
	"deepset-ai/haystack" : {
		labels : null,
	},
	"TransformerOptimus/SuperAGI" : {
		labels : null,
	},
	"letta-ai/letta" : { //Formally cpacker/MemGPT renamed to letta
		labels : null,
	},
	"langchain-ai/langchain" : {
		labels : null,
	}

};

export type RepoName = keyof typeof repoConfig;

// Change this before query
export const CURRENT_REPO: RepoName = "crewAIInc/crewAI";
interface SearchIssueResponseType {
	search: {
		nodes: Array<{
			number: number;
			title: string;
			body: string;
		}>;
		pageInfo: {
			endCursor: string | null;
			hasNextPage: boolean;
		};
		issueCount: number;
	};
}

function generateDateRanges(startYear = 2020, endYear = new Date().getFullYear()): Array<{start: string, end: string}> {
    const ranges: Array<{start: string, end: string}> = [];
    
    for (let year = startYear; year <= endYear; year++) {
        // Split each year into quarters to keep results under 1000
        const quarters = [
            { start: `${year}-01-01`, end: `${year}-03-31` },
            { start: `${year}-04-01`, end: `${year}-06-30` },
            { start: `${year}-07-01`, end: `${year}-09-30` },
            { start: `${year}-10-01`, end: `${year}-12-31` }
        ];
        ranges.push(...quarters);
    }
    
    return ranges;
}

async function getIssues() {
    const [owner, repo] = CURRENT_REPO.split("/");
    const _config = repoConfig[CURRENT_REPO];

    const outputDir = createDirNameFromRepo(CURRENT_REPO);
    const repoDir = join(ROOT_DIR, "data", outputDir);
    await Deno.mkdir(repoDir, { recursive: true });

    const jsonlFilePath = join(repoDir, "issues.jsonl");
    const file = await Deno.open(jsonlFilePath, {
        write: true,
        create: true,
        truncate: true,
    });
    const encoder = new TextEncoder();

    let totalIssues = 0;
    const dateRanges = generateDateRanges();
    const issueNumbers = new Set<number>(); // Track to avoid duplicates

    console.log(`Processing ${dateRanges.length} date ranges for ${CURRENT_REPO}...`);

    // Replace the single pageIterator with a loop through date ranges
    for (const range of dateRanges) {
        console.log(`\nProcessing date range: ${range.start} to ${range.end}`);
        
        const pageIterator =
            octokit.graphql.paginate.iterator<SearchIssueResponseType>(
                `query paginate($searchQuery: String!, $cursor: String) {
            search(query: $searchQuery, type: ISSUE, first: 100, after: $cursor) {
              nodes {
                ... on Issue {
                  number
                  title
                  body
                }
              }
              pageInfo {
                endCursor
                hasNextPage
              }
              issueCount
            }
          }`,
                {
                    searchQuery: `repo:${owner}/${repo} is:issue state:closed linked:pr reason:completed created:${range.start}..${range.end}`,
                },
            );

        let isFirstPage = true;
        let rangeIssues = 0;

        try {
            for await (const response of pageIterator) {
                const search = response.search;

                if (isFirstPage) {
                    console.log(`  Found ${search.issueCount} issues in this range`);
                    isFirstPage = false;
                }

                for (const issue of search.nodes) {
                    // Skip duplicates
                    if (issueNumbers.has(issue.number)) {
                        continue;
                    }
                    
                    issueNumbers.add(issue.number);
                    const jsonLine = `${JSON.stringify(issue)}\n`;
                    await file.write(encoder.encode(jsonLine));
                    console.log(`  Saved issue #${issue.number}: ${issue.title}`);
                    totalIssues++;
                    rangeIssues++;
                }
            }
        } catch (error) {
            console.error(`Error processing range ${range.start} to ${range.end}:`, error);
            // Continue with next range instead of failing completely
            continue;
        }

        console.log(`Completed range: ${rangeIssues} new issues saved`);
        
        // Add delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    file.close();
    console.log(`\n All ${totalIssues} issues saved to ${jsonlFilePath}`);
    
}

if (import.meta.main) {
	try {
		await getIssues();
	} catch (error) {
		console.error("Script failed with error:", error);
		if (error instanceof Error) {
			console.error("Error stack:", error.stack);
		}
	}
	Deno.exit(0);
}
