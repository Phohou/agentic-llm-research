import { join } from "@std/path";
import { ROOT_DIR } from "../utils/directory.ts";
import { getRepoNameFromDir } from "../utils/repo.ts";

interface CommentData {
	id: string;
	number: number;
	title: string;
	body: string;
	createdAt: string;
	updatedAt: string;
	closed: boolean;
	closedAt: string | null;
	author: {
		login: string;
	} | null;
	labels: {
		nodes: Array<{
			id: string;
			name: string;
		}>;
	};
	comments: Array<{
		id: string;
		body: string;
		createdAt: string;
		updatedAt: string;
		author: {
			login: string;
		} | null;
		reactionGroups: Array<{
			content: string;
			reactors: {
				totalCount: number;
			};
		}>;
	}>;
	reactionGroups: Array<{
		content: string;
		reactors: {
			totalCount: number;
		};
	}>;
	repository: {
		nameWithOwner: string;
	};
	closedByPullRequests?: {
		totalCount: number;
		nodes?: Array<{
			id: string;
			number: number;
		}>;
	};
}

interface CSVRow {
	repo: string;
	issue_number: number;
	issue_title: string;
	issue_url: string;
	number_of_comments: number;
	total_reactions_in_issue: number;
	total_reactions_in_comments: number;
	closed_by_pull_requests: number;
}

function countReactions(
	reactionGroups: Array<{ content: string; reactors: { totalCount: number } }>,
): number {
	return reactionGroups.reduce(
		(total, group) => total + group.reactors.totalCount,
		0,
	);
}

async function processRepository(repoDir: string): Promise<CSVRow[]> {
	const repoName = getRepoNameFromDir(repoDir);
	const commentsPath = join(ROOT_DIR, "data", repoDir, "comments.jsonl");

	const results: CSVRow[] = [];

	try {
		const commentsText = await Deno.readTextFile(commentsPath);
		const lines = commentsText.trim().split("\n");

		for (const line of lines) {
			if (line.trim()) {
				const commentData: CommentData = JSON.parse(line);

				const issueReactions = countReactions(commentData.reactionGroups);
				const commentReactions = commentData.comments.reduce(
					(total, comment) => total + countReactions(comment.reactionGroups),
					0,
				);

				// Create issue URL
				const issueUrl = `https://github.com/${repoName}/issues/${commentData.number}`;

				// Get number of PRs that closed this issue
				const closedByPRCount =
					commentData.closedByPullRequests?.totalCount || 0;

				results.push({
					repo: repoName,
					issue_number: commentData.number,
					issue_url: issueUrl,
					issue_title: commentData.title,
					number_of_comments: commentData.comments.length,
					total_reactions_in_issue: issueReactions,
					total_reactions_in_comments: commentReactions,
					closed_by_pull_requests: closedByPRCount,
				});
			}
		}
	} catch (error) {
		console.error(`Error processing repository ${repoDir}:`, error);
	}

	return results;
}

async function generateCSVReport(): Promise<void> {
	const dataDir = join(ROOT_DIR, "data");
	const allResults: CSVRow[] = [];

	try {
		for await (const dirEntry of Deno.readDir(dataDir)) {
			if (dirEntry.isDirectory) {
				console.log(`Processing repository: ${dirEntry.name}`);
				const repoResults = await processRepository(dirEntry.name);
				allResults.push(...repoResults);
			}
		}

		// Generate CSV content
		const csvHeader =
			"repo,issue_number,issue_url,issue_title,number_of_comments,total_reactions_in_issue,total_reactions_in_comments,closed_by_pull_requests\n";
		const csvRows = allResults
			.map(
				(row) =>
					`"${row.repo}",${row.issue_number},"${row.issue_url}","${row.issue_title.replace(
						/"/g,
						'""',
					)}",${row.number_of_comments},${row.total_reactions_in_issue},${
						row.total_reactions_in_comments
					},${row.closed_by_pull_requests}`,
			)
			.join("\n");

		const csvContent = csvHeader + csvRows;

		const outputPath = join(ROOT_DIR, "output", "issues_report.csv");
		await Deno.mkdir(join(ROOT_DIR, "output"), { recursive: true });
		await Deno.writeTextFile(outputPath, csvContent);

		console.log(`\nCSV report generated successfully!`);
		console.log(`Output file: ${outputPath}`);
		console.log(`Total issues processed: ${allResults.length}`);
	} catch (error) {
		console.error("Error generating CSV report:", error);
	}
}

if (import.meta.main) {
	await generateCSVReport();
}
