## rules

- **Never delete existing code** unless the user explicitly names a function/block to remove. Treat all existing code — including commented-out lines, alternative implementations, and exploratory cells — as intentional.
- **Always merge, never replace.** When adding new behavior, extend the existing code (new branch, new cell, additional column, optional parameter). Do not rewrite a function/cell from scratch when an additive change works.
- **Preserve structure on edits**: keep cell boundaries, existing imports, variable names, and ordering. Add new imports/constants alongside existing ones rather than reorganizing.
- If a change seems to require deletion or a rewrite, stop and ask first.
- Ran the affected notebook end-to-end (or stated you didn't and why).
- Leave spacing / whitespace as is, do not attempt to inline '=' over mutliple lines or create more whitespace
- always make changes directly to the python files, notebooks etc, never provide the code in the chat unless explicidly asked to do so



