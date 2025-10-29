import { Octokit } from "https://esm.sh/octokit";
import "jsr:@std/dotenv/load";
import { join } from "@std/path";
import { ROOT_DIR } from "../utils/directory.ts";
import { createDirNameFromRepo } from "../utils/repo.ts";
import type { RepoName } from "./get_issues.ts";

const octokit = new Octokit({ auth: Deno.env.get("GITHUB_TOKEN_1") });

// All available repositories
const ALL_REPOS: RepoName[] = [
	"run-llama/llama_index",
	"microsoft/autogen",
	"crewAIInc/crewAI",
	"microsoft/semantic-kernel",
	"deepset-ai/haystack", 
	"TransformerOptimus/SuperAGI",
	"letta-ai/letta",
	"langchain-ai/langchain"
];

interface PullRequest {
	number: number;
	title: string;
	additions: number;
	deletions: number;
	changedFiles: number;
}

interface IssueWithComments {
	id: string;
	number: number;
	title: string;
	body: string;
	createdAt: string;
	updatedAt: string;
	closed: boolean;
	closedAt: string | null;
	author: { login: string } | null;
	labels: {
		nodes: Array<{
			id: string;
			name: string;
		}>;
	};
	reactionGroups: Array<{
		content: string;
		reactors: {
			totalCount: number;
		};
	}>;
	comments: Array<{
		id: string;
		body: string;
		createdAt: string;
		updatedAt: string;
		author: { login: string } | null;
		reactionGroups: Array<{
			content: string;
			reactors: {
				totalCount: number;
			};
		}>;
	}>;
	repository: {
		nameWithOwner: string;
	};
	closedByPullRequests?: PullRequest[];
}

interface RepositoryIssueClosingPRResponseType {
	repository: {
		issue: {
			closedByPullRequestsReferences: {
				nodes: Array<{
					number: number;
					title: string;
				}>;
				pageInfo: {
					endCursor: string | null;
					hasNextPage: boolean;
				};
				totalCount: number;
			};
		};
	};
}

async function readCommentsFromFile(
	repoName: RepoName,
): Promise<IssueWithComments[]> {
	const outputDir = createDirNameFromRepo(repoName);
	const repoDir = join(ROOT_DIR, "data", outputDir);
	const jsonlFilePath = join(repoDir, "comments.jsonl");

	try {
		const content = await Deno.readTextFile(jsonlFilePath);
		const issues: IssueWithComments[] = content
			.trim()
			.split("\n")
			.filter((line) => line.trim())
			.map((line) => JSON.parse(line));

		return issues;
	} catch (error) {
		console.error(`Error reading comments file for ${repoName}: ${error}`);
		throw error;
	}
}

async function getClosingPRsForIssue(
	repoName: RepoName,
	issueNumber: number,
): Promise<PullRequest[]> {
	const [owner, repo] = repoName.split("/");

	const pageIterator =
		octokit.graphql.paginate.iterator<RepositoryIssueClosingPRResponseType>(
			`query paginate($owner: String!, $repo: String!, $issueNumber: Int!, $cursor: String) {
        repository(owner: $owner, name: $repo) {
          issue(number: $issueNumber) {
            closedByPullRequestsReferences(first: 100, after: $cursor) {
              nodes {
                number
                title
              }
              pageInfo {
                endCursor
                hasNextPage
              }
              totalCount
            }
          }
        }
      }`,
			{
				owner,
				repo,
				issueNumber,
			},
		);

	const allPRs: PullRequest[] = [];

	try {
		for await (const response of pageIterator) {
			const prs =
				response.repository.issue.closedByPullRequestsReferences.nodes;
			allPRs.push(...prs);
		}
	} catch (error) {
		console.error(
			`Error fetching closing PRs for issue #${issueNumber}:`,
			error,
		);
		// Return empty array if there's an error, to avoid breaking the whole process
		return [];
	}

	return allPRs;
}

async function updateCommentsWithClosingPRs(repoName: RepoName) {
	console.log(`\n=== Processing ${repoName} ===`);
	console.log("Reading issues with comments from file...");
	const issues = await readCommentsFromFile(repoName);
	console.log(`Successfully read ${issues.length} issues from file`);

	console.log(`Processing all ${issues.length} issues to get closing PRs`);

	const outputDir = createDirNameFromRepo(repoName);
	const repoDir = join(ROOT_DIR, "data", outputDir);
	await Deno.mkdir(repoDir, { recursive: true });

	const jsonlFilePath = join(repoDir, "comments.jsonl");
	const tempFilePath = join(repoDir, "comments.jsonl.tmp");
	const backupFilePath = join(repoDir, "comments.jsonl.backup");

	// Create backup of original file
	try {
		await Deno.copyFile(jsonlFilePath, backupFilePath);
		console.log(`Created backup: ${backupFilePath}`);
	} catch (error) {
		console.error(`Warning: Could not create backup: ${error}`);
	}

	const file = await Deno.open(tempFilePath, {
		write: true,
		create: true,
		truncate: true,
	});
	const encoder = new TextEncoder();

	let totalPRs = 0;

	for (const issue of issues) {
		console.log(
			`Fetching closing PRs for issue #${issue.number}: ${issue.title}`,
		);

		try {
			const closingPRs = await getClosingPRsForIssue(repoName, issue.number);

			// Update the issue with closing PR information
			const updatedIssue: IssueWithComments = {
				...issue,
				closedByPullRequests: closingPRs,
			};

			const jsonLine = `${JSON.stringify(updatedIssue)}\n`;
			await file.write(encoder.encode(jsonLine));
			totalPRs += closingPRs.length;

			if (closingPRs.length > 0) {
				console.log(
					`  Found ${closingPRs.length} closing PR(s) for issue #${issue.number}:`,
				);
				closingPRs.forEach((pr) => {
					console.log(`    - PR #${pr.number}: ${pr.title}`);
				});
			} else {
				console.log(`  No closing PRs found for issue #${issue.number}`);
			}

			// Add a longer delay to avoid hitting rate limits
			await new Promise((resolve) => setTimeout(resolve, 1000)); // 1 second delay
		} catch (error) {
			console.error(`Error processing issue #${issue.number}: ${error}`);

			// Still write the original issue data even if PR fetching failed
			const issueWithoutPRs: IssueWithComments = {
				...issue,
				closedByPullRequests: [],
			};

			const jsonLine = `${JSON.stringify(issueWithoutPRs)}\n`;
			await file.write(encoder.encode(jsonLine));
		}
	}

	file.close();

	// If we got here, the temp file was written successfully
	// Now move the temp file to replace the original
	try {
		await Deno.rename(tempFilePath, jsonlFilePath);
		console.log(`Successfully updated ${jsonlFilePath}`);
	} catch (error) {
		console.error(`Error moving temp file to final location: ${error}`);
		// Try to restore from backup if available
		try {
			await Deno.copyFile(backupFilePath, jsonlFilePath);
			console.log(`Restored original file from backup`);
		} catch (restoreError) {
			console.error(`Failed to restore from backup: ${restoreError}`);
		}
		throw error;
	}

	console.log(
		`Updated ${issues.length} issues with ${totalPRs} total closing PRs`,
	);
	console.log(`Updated data saved to ${jsonlFilePath}`);
}

if (import.meta.main) {
	try {
		console.log("Starting to process all repositories for closing PRs...");

		for (const repo of ALL_REPOS) {
			try {
				await updateCommentsWithClosingPRs(repo);
			} catch (error) {
				console.error(`Failed to process ${repo}:`, error);
				console.log(`Continuing with next repository...`);
			}
		}

		console.log("All repositories processed successfully");
	} catch (error) {
		console.error("Script failed with error:", error);
		if (error instanceof Error) {
			console.error("Error stack:", error.stack);
		}
	}
	Deno.exit(0);
}
