import { assert } from "@std/assert/assert";

export function createDirNameFromRepo(repo_name: string): string {
	assert(
		/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$/.test(repo_name),
		"Repo name must be in the format 'owner/repo'.",
	);
	return repo_name.replace("/", "+");
}

export function getRepoNameFromDir(dir: string): string {
	assert(
		/^[a-zA-Z0-9_-]+\+[a-zA-Z0-9_-]+$/.test(dir),
		"Directory name must be in the format 'owner+repo'.",
	);
	return dir.replace("+", "/");
}

if (import.meta.main) {
	const repoName = "libquantum/cirq";
	const dirName = createDirNameFromRepo(repoName);
	console.log("Directory name for repo:", dirName);

	const extractedRepoName = getRepoNameFromDir("libquantum+cirq");
	console.log("Extracted repo name from directory:", extractedRepoName);
}
