import * as childProcess from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";


const PCAT_VIEW_TYPE = "pcat.viewer";
const PCAT_PACKAGE_SOURCE = "git+https://github.com/anhvth/speedy_utils";


type PromptedLaunchOptions = {
    index?: string;
    split?: string;
    plain?: boolean;
    sample?: string;
};

type PcatConfig = {
    command: string;
    extraArgs: string[];
    defaultSplit: string;
    terminalName: string;
    reuseTerminal: boolean;
    autoLaunchOnOpen: boolean;
};


type PcatReadonlyDocument = vscode.CustomDocument & {
    hasLaunched: boolean;
};


let sharedTerminal: vscode.Terminal | undefined;


export function activate(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.window.registerCustomEditorProvider(
            PCAT_VIEW_TYPE,
            new PcatReadonlyEditorProvider(context),
            {
                supportsMultipleEditorsPerDocument: true,
                webviewOptions: {
                    retainContextWhenHidden: true,
                },
            },
        ),
        vscode.commands.registerCommand("pcat.open", async (arg?: unknown) => {
            await runInteractive(arg, { askForIndex: false });
        }),
        vscode.commands.registerCommand("pcat.openAtIndex", async (arg?: unknown) => {
            await runInteractive(arg, { askForIndex: true });
        }),
        vscode.commands.registerCommand(
            "pcat.previewPlainRow",
            async (arg?: unknown) => {
                await runPlainPreview(arg);
            },
        ),
        vscode.commands.registerCommand("pcat.reopenAsText", async (arg?: unknown) => {
            const targetUri = await resolveTargetUri(arg, "Reopen as text");
            if (!targetUri) {
                return;
            }

            await vscode.commands.executeCommand(
                "vscode.openWith",
                targetUri,
                "default",
            );
        }),
        vscode.window.onDidCloseTerminal((terminal) => {
            if (terminal === sharedTerminal) {
                sharedTerminal = undefined;
            }
        }),
    );
}


export function deactivate(): void {
    return;
}


class PcatReadonlyEditorProvider
    implements vscode.CustomReadonlyEditorProvider<PcatReadonlyDocument> {
    constructor(private readonly context: vscode.ExtensionContext) {
        void this.context.extensionMode;
    }

    openCustomDocument(uri: vscode.Uri): PcatReadonlyDocument {
        return {
            uri,
            hasLaunched: false,
            dispose: () => undefined,
        };
    }

    async resolveCustomEditor(
        document: PcatReadonlyDocument,
        webviewPanel: vscode.WebviewPanel,
    ): Promise<void> {
        webviewPanel.webview.options = {
            enableScripts: true,
        };
        webviewPanel.title = `${path.basename(document.uri.fsPath)} · PCAT`;
        webviewPanel.webview.html = getCustomEditorHtml(webviewPanel.webview, document.uri);

        webviewPanel.webview.onDidReceiveMessage(
            async (message: { type?: string }) => {
                if (message.type === "open") {
                    await openInteractiveForUri(document.uri, {});
                    return;
                }

                if (message.type === "openAtIndex") {
                    const index = await promptForIndex();
                    if (index === undefined) {
                        return;
                    }
                    await openInteractiveForUri(document.uri, { index });
                    return;
                }

                if (message.type === "preview") {
                    const index = await promptForIndex();
                    if (index === undefined) {
                        return;
                    }
                    await previewPlainForUri(document.uri, { index });
                    return;
                }

                if (message.type === "sample") {
                    const sample = await promptForSample();
                    if (sample === undefined) {
                        return;
                    }
                    await openInteractiveForUri(document.uri, { sample });
                    return;
                }

                if (message.type === "text") {
                    await vscode.commands.executeCommand(
                        "vscode.openWith",
                        document.uri,
                        "default",
                    );
                }
            },
            undefined,
            this.context.subscriptions,
        );

        if (!document.hasLaunched && readConfig().autoLaunchOnOpen) {
            document.hasLaunched = true;
            await openInteractiveForUri(document.uri, {});
        }
    }
}


async function runInteractive(
    arg: unknown,
    options: { askForIndex: boolean },
): Promise<void> {
    const targetUri = await resolveTargetUri(arg, "Open with PCAT");
    if (!targetUri) {
        return;
    }

    const config = readConfig();
    const launchOptions = await collectLaunchOptions(targetUri, config, {
        askForIndex: options.askForIndex,
    });
    if (!launchOptions) {
        return;
    }

    await openInteractiveForUri(targetUri, launchOptions);
}


async function runPlainPreview(arg: unknown): Promise<void> {
    const targetUri = await resolveTargetUri(arg, "Preview plain row with PCAT");
    if (!targetUri) {
        return;
    }

    const config = readConfig();
    const launchOptions = await collectLaunchOptions(targetUri, config, {
        askForIndex: true,
    });
    if (!launchOptions) {
        return;
    }

    await previewPlainForUri(targetUri, launchOptions);
}


async function resolveTargetUri(
    arg: unknown,
    openLabel: string,
): Promise<vscode.Uri | undefined> {
    const candidate = extractUri(arg) ?? vscode.window.activeTextEditor?.document.uri;
    if (candidate) {
        return validateTargetUri(candidate);
    }

    const picked = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: true,
        canSelectMany: false,
        openLabel,
    });
    return validateTargetUri(picked?.[0]);
}


function extractUri(arg: unknown): vscode.Uri | undefined {
    if (arg instanceof vscode.Uri) {
        return arg;
    }

    if (Array.isArray(arg) && arg[0] instanceof vscode.Uri) {
        return arg[0];
    }

    if (
        typeof arg === "object" &&
        arg !== null &&
        "resourceUri" in arg &&
        arg.resourceUri instanceof vscode.Uri
    ) {
        return arg.resourceUri;
    }

    return undefined;
}


function validateTargetUri(
    uri: vscode.Uri | undefined,
): vscode.Uri | undefined {
    if (!uri) {
        return undefined;
    }

    if (uri.scheme !== "file") {
        void vscode.window.showErrorMessage("PCAT only supports local file system paths.");
        return undefined;
    }

    if (!fs.existsSync(uri.fsPath)) {
        void vscode.window.showErrorMessage(`Path not found: ${uri.fsPath}`);
        return undefined;
    }

    return uri;
}


async function collectLaunchOptions(
    targetUri: vscode.Uri,
    config: PcatConfig,
    options: { askForIndex: boolean },
): Promise<PromptedLaunchOptions | undefined> {
    let index: string | undefined;
    if (options.askForIndex) {
        const prompted = await promptForIndex();
        if (prompted === undefined) {
            return undefined;
        }
        index = prompted;
    }

    let split = normalizeOptional(config.defaultSplit);
    if (looksLikeHfDataset(targetUri.fsPath)) {
        const promptedSplit = await promptForSplit(split);
        if (promptedSplit === undefined) {
            return undefined;
        }
        split = promptedSplit;
    }

    return {
        index,
        split,
    };
}


async function promptForIndex(): Promise<string | undefined> {
    const value = await vscode.window.showInputBox({
        prompt: "Zero-based row index. Leave empty to open the last row. Negative values count from the end.",
        placeHolder: "Example: 0, 10, -1",
        validateInput: (input) => {
            const trimmed = input.trim();
            if (!trimmed || /^-?\d+$/.test(trimmed)) {
                return null;
            }
            return "Enter an integer or leave it empty.";
        },
    });
    if (value === undefined) {
        return undefined;
    }
    return normalizeOptional(value);
}


async function promptForSplit(
    currentValue: string | undefined,
): Promise<string | undefined> {
    const value = await vscode.window.showInputBox({
        prompt: "Dataset split for DatasetDict inputs. Leave empty to let pcat decide.",
        placeHolder: currentValue || "train",
        value: currentValue,
    });
    if (value === undefined) {
        return undefined;
    }
    return normalizeOptional(value);
}


async function promptForSample(): Promise<string | undefined> {
    const value = await vscode.window.showInputBox({
        prompt: "Number of random rows to sample. Leave empty for 1.",
        placeHolder: "Example: 10, 100",
        validateInput: (input) => {
            const trimmed = input.trim();
            if (!trimmed || /^\d+$/.test(trimmed)) {
                return null;
            }
            return "Enter a positive integer or leave it empty.";
        },
    });
    if (value === undefined) {
        return undefined;
    }
    const trimmed = value.trim();
    return trimmed || "1";
}


function normalizeOptional(value: string | undefined): string | undefined {
    const trimmed = value?.trim();
    return trimmed ? trimmed : undefined;
}


function looksLikeHfDataset(targetPath: string): boolean {
    try {
        const stat = fs.statSync(targetPath);
        return stat.isDirectory() && fs.existsSync(path.join(targetPath, "dataset_info.json"));
    } catch {
        return false;
    }
}


function readConfig(): PcatConfig {
    const config = vscode.workspace.getConfiguration("pcat");
    return {
        command: config.get<string>("command", "pcat"),
        extraArgs: config.get<string[]>("extraArgs", []),
        defaultSplit: config.get<string>("defaultSplit", ""),
        terminalName: config.get<string>("terminalName", "PCAT"),
        reuseTerminal: config.get<boolean>("reuseTerminal", true),
        autoLaunchOnOpen: config.get<boolean>("autoLaunchOnOpen", true),
    };
}


async function openInteractiveForUri(
    targetUri: vscode.Uri,
    launchOptions: PromptedLaunchOptions,
): Promise<void> {
    try {
        const config = await ensurePcatCommand(readConfig());
        const terminal = getTerminal(config, determineCwd(targetUri));
        const cwd = determineCwd(targetUri);
        if (cwd) {
            terminal.sendText(`cd ${quoteForShell(cwd)}`, true);
        }

        terminal.sendText(buildCommandLine(config, targetUri.fsPath, launchOptions), true);
        terminal.show();
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`PCAT launch failed: ${message}`);
    }
}


async function previewPlainForUri(
    targetUri: vscode.Uri,
    launchOptions: PromptedLaunchOptions,
): Promise<void> {
    try {
        const config = await ensurePcatCommand(readConfig());
        const commandLine = buildCommandLine(config, targetUri.fsPath, {
            ...launchOptions,
            plain: true,
        });
        const output = await execCommand(commandLine, determineCwd(targetUri));
        const content = output.trim();
        if (!content) {
            vscode.window.showWarningMessage("pcat returned no output.");
            return;
        }

        const document = await vscode.workspace.openTextDocument({
            language: "json",
            content: `${content}\n`,
        });
        await vscode.window.showTextDocument(document, { preview: false });
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`PCAT preview failed: ${message}`);
    }
}


async function ensurePcatCommand(config: PcatConfig): Promise<PcatConfig> {
    if (config.command.trim() !== "pcat") {
        return config;
    }

    const existingCommand = await findExecutableOnPath("pcat");
    if (existingCommand) {
        return config;
    }

    const existingUvToolCommand = await findUvToolExecutable("pcat");
    if (existingUvToolCommand) {
        return {
            ...config,
            command: quoteForShell(existingUvToolCommand),
        };
    }

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: "Installing pcat",
            cancellable: false,
        },
        async () => {
            await execCommand(`uv tool install ${PCAT_PACKAGE_SOURCE}`, undefined);
        },
    );

    const installedCommand = await findExecutableOnPath("pcat");
    if (installedCommand) {
        return config;
    }

    const uvToolCommand = await findUvToolExecutable("pcat");
    if (uvToolCommand) {
        return {
            ...config,
            command: quoteForShell(uvToolCommand),
        };
    }

    throw new Error(
        "Installed pcat with uv, but the pcat executable was not found. Run `uv tool dir --bin` and add that directory to PATH.",
    );
}


async function findExecutableOnPath(command: string): Promise<string | undefined> {
    try {
        const checkCommand = process.platform === "win32"
            ? `where ${quoteForWindowsShell(command)}`
            : `command -v ${quoteForShell(command)}`;
        const output = await execCommand(checkCommand, undefined);
        return normalizeOptional(output.split(/\r?\n/, 1)[0]);
    } catch {
        return undefined;
    }
}


async function findUvToolExecutable(command: string): Promise<string | undefined> {
    try {
        const output = await execCommand("uv tool dir --bin", undefined);
        const binDir = normalizeOptional(output);
        if (!binDir) {
            return undefined;
        }

        const executable = path.join(
            binDir,
            process.platform === "win32" ? `${command}.exe` : command,
        );
        return fs.existsSync(executable) ? executable : undefined;
    } catch {
        return undefined;
    }
}


function getTerminal(config: PcatConfig, cwd: string | undefined): vscode.Terminal {
    if (config.reuseTerminal && sharedTerminal) {
        return sharedTerminal;
    }

    const terminal = vscode.window.createTerminal({
        name: config.terminalName,
        cwd,
    });

    if (config.reuseTerminal) {
        sharedTerminal = terminal;
    }

    return terminal;
}


function buildCommandLine(
    config: PcatConfig,
    targetPath: string,
    options: PromptedLaunchOptions,
): string {
    const command = config.command.trim();
    if (!command) {
        throw new Error("The pcat.command setting is empty.");
    }

    const args = [...config.extraArgs];
    if (options.plain) {
        args.push("--plain");
    }
    if (options.sample) {
        args.push("--sample", options.sample);
    } else if (options.index) {
        args.push("--index", options.index);
    }
    if (options.split) {
        args.push("--split", options.split);
    }
    args.push(targetPath);

    return [command, ...args.map((arg) => quoteForShell(arg))].join(" ");
}


function determineCwd(targetUri: vscode.Uri): string | undefined {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(targetUri);
    if (workspaceFolder) {
        return workspaceFolder.uri.fsPath;
    }

    try {
        const stat = fs.statSync(targetUri.fsPath);
        return stat.isDirectory() ? targetUri.fsPath : path.dirname(targetUri.fsPath);
    } catch {
        return path.dirname(targetUri.fsPath);
    }
}


function quoteForShell(value: string): string {
    if (process.platform === "win32") {
        return quoteForWindowsShell(value);
    }

    if (!value) {
        return "''";
    }

    return `'${value.replace(/'/g, `'\\''`)}'`;
}


function quoteForWindowsShell(value: string): string {
    return `"${value.replace(/"/g, '\\"')}"`;
}


function execCommand(
    command: string,
    cwd: string | undefined,
): Promise<string> {
    return new Promise((resolve, reject) => {
        childProcess.exec(
            command,
            {
                cwd,
                maxBuffer: 10 * 1024 * 1024,
            },
            (
                error: childProcess.ExecException | null,
                stdout: string,
                stderr: string,
            ) => {
                if (error) {
                    const message = stderr.trim() || stdout.trim() || error.message;
                    reject(new Error(message));
                    return;
                }
                resolve(stdout);
            },
        );
    });
}


function getCustomEditorHtml(webview: vscode.Webview, targetUri: vscode.Uri): string {
    const nonce = getNonce();
    const escapedPath = escapeHtml(targetUri.fsPath);
    const escapedName = escapeHtml(path.basename(targetUri.fsPath));

    return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapedName}</title>
    <style>
      :root {
        color-scheme: light dark;
      }

      body {
        font-family: var(--vscode-font-family);
        padding: 24px;
        color: var(--vscode-foreground);
        background: linear-gradient(180deg, var(--vscode-editor-background) 0%, color-mix(in srgb, var(--vscode-editor-background) 92%, var(--vscode-textLink-foreground) 8%) 100%);
      }

      .card {
        max-width: 760px;
        border: 1px solid var(--vscode-panel-border);
        background: color-mix(in srgb, var(--vscode-editor-background) 88%, var(--vscode-textLink-foreground) 12%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.12);
      }

      h1 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.4rem;
      }

      p {
        line-height: 1.5;
      }

      code {
        font-family: var(--vscode-editor-font-family);
      }

      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 20px;
      }

      button {
        border: none;
        border-radius: 999px;
        padding: 10px 16px;
        cursor: pointer;
        color: var(--vscode-button-foreground);
        background: var(--vscode-button-background);
      }

      button.secondary {
        color: var(--vscode-button-secondaryForeground);
        background: var(--vscode-button-secondaryBackground);
      }

      .path {
        padding: 12px;
        border-radius: 10px;
        background: color-mix(in srgb, var(--vscode-editor-background) 76%, black 24%);
        overflow-wrap: anywhere;
      }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>${escapedName}</h1>
      <p>This .jsonl file is associated with the PCAT custom editor. PCAT launches in the integrated terminal for interactive navigation.</p>
      <div class="path"><code>${escapedPath}</code></div>
      <div class="actions">
        <button data-action="open">Open PCAT</button>
        <button data-action="openAtIndex">Open At Row</button>
        <button data-action="sample">Sample Rows</button>
        <button data-action="preview" class="secondary">Preview Plain Row</button>
        <button data-action="text" class="secondary">Open As Text</button>
      </div>
    </div>
    <script nonce="${nonce}">
      const vscode = acquireVsCodeApi();
      for (const button of document.querySelectorAll('button[data-action]')) {
        button.addEventListener('click', () => {
          vscode.postMessage({ type: button.dataset.action });
        });
      }
    </script>
  </body>
</html>`;
}


function escapeHtml(value: string): string {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}


function getNonce(): string {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let value = "";
    for (let i = 0; i < 32; i += 1) {
        value += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return value;
}
