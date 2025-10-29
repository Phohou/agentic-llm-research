import { Octokit } from "https://esm.sh/octokit";
import "jsr:@std/dotenv/load";
import { join } from "@std/path";
import { ROOT_DIR } from "../utils/directory.ts";
import { createDirNameFromRepo } from "../utils/repo.ts";
import { CURRENT_REPO } from "./get_issues.ts";

// import { type RepoName } from "./get_issues.ts";

const octokit = new Octokit({ auth: Deno.env.get("GITHUB_TOKEN_1") });

// Change this before query to match the repo you want to get comments for
// const CURRENT_REPO: RepoName = "pasqal-io/emulators";

// Simple issue structure from issues.jsonl file
interface IssueFromFile {
	number: number;
	title: string;
}

interface IssueCore {
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
}

interface Issue extends IssueCore {
	comments: {
		totalCount: number;
	};
	repository: {
		nameWithOwner: string;
	};
}

interface Comment {
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
}

interface RepositoryIssueCommentsResponseType {
	repository: {
		nameWithOwner: string;
		issue: IssueCore & {
			comments: {
				nodes: Comment[];
				pageInfo: {
					endCursor: string | null;
					hasNextPage: boolean;
				};
				totalCount: number;
			};
		};
	};
}

async function readIssuesFromFile(): Promise<IssueFromFile[]> {
	const outputDir = createDirNameFromRepo(CURRENT_REPO);
	const repoDir = join(ROOT_DIR, "data", outputDir);
	const jsonlFilePath = join(repoDir, "issues.jsonl");

	try {
		const content = await Deno.readTextFile(jsonlFilePath);
		const issues: IssueFromFile[] = content
			.trim()
			.split("\n")
			.filter((line) => line.trim())
			.map((line) => JSON.parse(line));

		return issues;
	} catch (error) {
		console.error(`Error reading issues file: ${error}`);
		throw error;
	}
}

async function getCommentsForIssue(
	issueNumber: number,
): Promise<{ issue: Issue; comments: Comment[] }> {
	const [owner, repo] = CURRENT_REPO.split("/");

	const pageIterator =
		octokit.graphql.paginate.iterator<RepositoryIssueCommentsResponseType>(
			`query paginate($owner: String!, $repo: String!, $issueNumber: Int!, $cursor: String) {
        repository(owner: $owner, name: $repo) {
          nameWithOwner
          issue(number: $issueNumber) {
            id
            number
            title
            body
            createdAt
            updatedAt
            closed
            closedAt
            author {
              login
            }
            labels(first: 100) {
              nodes {
                id
                name
              }
            }
            reactionGroups {
              content
              reactors {
                totalCount
              }
            }
            comments(first: 100, after: $cursor) {
              nodes {
                id
                body
                createdAt
                updatedAt
                author {
                  login
                }
                reactionGroups {
                  content
                  reactors {
                    totalCount
                  }
                }
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

	const allComments: Comment[] = [];
	let issueData: Issue | null = null;
	let repoNameWithOwner: string | null = null;

	for await (const response of pageIterator) {
		const repository = response.repository;
		const issue = repository.issue;

		// Store repository name from the first response
		if (!repoNameWithOwner) {
			repoNameWithOwner = repository.nameWithOwner;
		}

		// Store issue data from the first response
		if (!issueData) {
			issueData = {
				id: issue.id,
				number: issue.number,
				title: issue.title,
				body: issue.body,
				createdAt: issue.createdAt,
				updatedAt: issue.updatedAt,
				closed: issue.closed,
				closedAt: issue.closedAt,
				author: issue.author,
				labels: issue.labels,
				// The comments totalCount is part of IssueCore, but we need to ensure it's set correctly
				comments: { totalCount: issue.comments.totalCount },
				reactionGroups: issue.reactionGroups,
				repository: {
					nameWithOwner: repoNameWithOwner,
				},
			};
		}

		allComments.push(...issue.comments.nodes);
	}

	// Assert that we got issue data - this should never be null if the API returned any responses
	if (!issueData) {
		throw new Error(
			`Failed to fetch issue data for issue #${issueNumber}. No responses received from GitHub API.`,
		);
	}

	return {
		issue: issueData,
		comments: allComments,
	};
}

async function getComments() {
	console.log("Reading issues from file...");
	const issues = await readIssuesFromFile();
	console.log(`Successfully read ${issues.length} issues from file`);

	console.log(
		`Processing all ${issues.length} issues (including those without comments)`,
	);

	const outputDir = createDirNameFromRepo(CURRENT_REPO);
	const repoDir = join(ROOT_DIR, "data", outputDir);
	await Deno.mkdir(repoDir, { recursive: true });

	const jsonlFilePath = join(repoDir, "comments.jsonl");
	const file = await Deno.open(jsonlFilePath, {
		write: true,
		create: true,
		truncate: true,
	});
	const encoder = new TextEncoder();

	let totalComments = 0;

	for (const issue of issues) {
		console.log(`Fetching comments for issue #${issue.number}: ${issue.title}`);

		try {
			const result = await getCommentsForIssue(issue.number);
			const { issue: fetchedIssue, comments } = result;

			// Create one entry per issue with issue data at root level and comments array
			const issueWithComments = {
				...fetchedIssue,
				comments: comments,
			};

			const jsonLine = `${JSON.stringify(issueWithComments)}\n`;
			await file.write(encoder.encode(jsonLine));
			totalComments += comments.length;

			console.log(
				`  Saved ${comments.length} comments for issue #${issue.number}`,
			);

			// Add a small delay to avoid hitting rate limits
			await new Promise((resolve) => setTimeout(resolve, 100));
		} catch (error) {
			console.error(
				`Error fetching comments for issue #${issue.number}: ${error}`,
			);
		}
	}

	file.close();
	console.log(`All ${totalComments} comments saved to ${jsonlFilePath}`);
}

if (import.meta.main) {
	try {
		await getComments();
		console.log("Script completed successfully");
	} catch (error) {
		console.error("Script failed with error:", error);
		if (error instanceof Error) {
			console.error("Error stack:", error.stack);
		}
	}
	Deno.exit(0);
}
