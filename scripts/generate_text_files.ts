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
}

function formatIssueText(data: CommentData, repoName: string): string {
	const labels = data.labels.nodes.map((label) => label.name).join(", ");
	const status = data.closed ? "closed" : "open";
	const authorName = data.author?.login || "unknown";

	let text = `#${data.number}: ${data.title}\n\n`;
	text += `repo: ${repoName}\n`;
	text += `author: ${authorName}\n`;
	text += `status: ${status}\n`;
	text += `labels: ${labels || "none"}\n\n`;
	text += `${data.body}\n\n`;

	if (data.comments.length > 0) {
		text += `---\nComments\n---\n\n`;

		data.comments.forEach((comment, index) => {
			const commentAuthor = comment.author?.login || "unknown";
			text += `#${index + 1}: ${comment.body} — ${commentAuthor}\n`;
			text += `---\n`;
		});
	}

	return text;
}

async function processRepository(repoDir: string): Promise<void> {
	const repoName = getRepoNameFromDir(repoDir);
	const commentsPath = join(ROOT_DIR, "data", repoDir, "comments.jsonl");
	const outputDir = join(ROOT_DIR, "output", "issues_text", repoDir);

	try {
		await Deno.mkdir(outputDir, { recursive: true });

		const commentsText = await Deno.readTextFile(commentsPath);
		const lines = commentsText.trim().split("\n");

		let processedCount = 0;

		for (const line of lines) {
			if (line.trim()) {
				const commentData: CommentData = JSON.parse(line);

				const issueText = formatIssueText(commentData, repoName);
				const fileName = `issue_${commentData.number}.txt`;
				const filePath = join(outputDir, fileName);

				await Deno.writeTextFile(filePath, issueText);
				processedCount++;
			}
		}

		console.log(
			`Processed ${processedCount} issues for repository: ${repoName}`,
		);
	} catch (error) {
		console.error(`Error processing repository ${repoDir}:`, error);
	}
}

async function generateTextFiles(): Promise<void> {
	const dataDir = join(ROOT_DIR, "data");
	let totalIssues = 0;

	try {
		console.log("Generating text files for all issues...\n");

		for await (const dirEntry of Deno.readDir(dataDir)) {
			if (dirEntry.isDirectory) {
				console.log(`Processing repository: ${dirEntry.name}`);
				await processRepository(dirEntry.name);

				// Count issues in this repository
				const commentsPath = join(dataDir, dirEntry.name, "comments.jsonl");
				try {
					const commentsText = await Deno.readTextFile(commentsPath);
					const lines = commentsText
						.trim()
						.split("\n")
						.filter((line) => line.trim());
					totalIssues += lines.length;
				} catch (error) {
					console.error(`Error counting issues in ${dirEntry.name}:`, error);
				}
			}
		}

		console.log(`\nText files generated successfully!`);
		console.log(`Output directory: ${join(ROOT_DIR, "output", "issues_text")}`);
		console.log(`Total issues processed: ${totalIssues}`);
	} catch (error) {
		console.error("Error generating text files:", error);
	}
}

if (import.meta.main) {
	await generateTextFiles();
}
