/**
 * LLM Call Logger - Records LLM input/output for debugging purposes.
 *
 * Configuration (in order of precedence):
 * 1. Environment variable: PI_LLM_LOG_DIR
 * 2. Settings: llmLogger.dir from settings.json
 * 3. Default: ~/.pi/llm-logs
 *
 * Enable via settings.json:
 * ```json
 * {
 *   "llmLogger": {
 *     "enabled": true,
 *     "dir": "/path/to/logs"
 *   }
 * }
 * ```
 */

import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { AssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { LlmLoggerSettings } from "./settings-manager.ts";

export interface LlmLogEntry {
	timestamp: string;
	callId: string;
	provider: string;
	modelId: string;
	request: {
		system?: string;
		messages: unknown[];
		options?: unknown;
	};
	response?: {
		id?: string;
		model?: string;
		role: string;
		content: unknown;
		usage?: unknown;
		stopReason?: string;
		error?: string;
	};
	durationMs: number;
}

export interface LlmLoggerConfig {
	enabled: boolean;
	dir: string;
}

class LlmLogger {
	private config: LlmLoggerConfig;
	private initialized = false;

	constructor() {
		// Default config, will be updated by init() or updateFromSettings()
		this.config = {
			enabled: false,
			dir: join(homedir(), ".pi", "llm-logs"),
		};
	}

	/**
	 * Initialize from environment variable or defaults.
	 * Called automatically during first use.
	 */
	private ensureInitialized(): void {
		if (this.initialized) return;
		this.initialized = true;

		// Environment variable takes highest priority
		const envDir = process.env.PI_LLM_LOG_DIR;
		if (envDir) {
			this.config = {
				enabled: true,
				dir: envDir,
			};
		} else if (!existsSync(this.config.dir)) {
			// Create default directory if it doesn't exist
			mkdirSync(this.config.dir, { recursive: true });
		}
	}

	/**
	 * Update configuration from settings.
	 * Can be called at runtime to apply settings changes.
	 */
	updateFromSettings(settings: LlmLoggerSettings | undefined): void {
		if (!settings) {
			// If settings is undefined/empty, reset to env or default
			this.initialized = false;
			this.ensureInitialized();
			return;
		}

		const enabled = settings.enabled ?? false;
		const dir = settings.dir ?? join(homedir(), ".pi", "llm-logs");

		this.config = { enabled, dir };

		// If enabling, ensure directory exists
		if (enabled && !existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}

		this.initialized = true;
	}

	isEnabled(): boolean {
		this.ensureInitialized();
		return this.config.enabled;
	}

	getLogDir(): string {
		this.ensureInitialized();
		return this.config.dir;
	}

	private generateCallId(): string {
		return `llm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
	}

	private sanitizeForLog(obj: unknown): unknown {
		if (obj === null || obj === undefined) return obj;
		if (typeof obj !== "object") return obj;

		// Clone to avoid mutating original objects
		const clone = Array.isArray(obj) ? [...obj] : { ...(obj as Record<string, unknown>) };

		for (const [key, value] of Object.entries(clone)) {
			if (key.toLowerCase().includes("key") || key.toLowerCase().includes("token")) {
				(clone as Record<string, unknown>)[key] = "[REDACTED]";
			} else if (typeof value === "object") {
				(clone as Record<string, unknown>)[key] = this.sanitizeForLog(value);
			}
		}

		return clone;
	}

	private extractSystemPrompt(context: unknown): { system?: string; messages: unknown[] } {
		if (!context || typeof context !== "object") {
			return { messages: [] };
		}

		const ctx = context as Record<string, unknown>;

		// Handle different context formats
		if (ctx.systemPrompt !== undefined) {
			return {
				system: typeof ctx.systemPrompt === "string" ? ctx.systemPrompt : JSON.stringify(ctx.systemPrompt),
				messages: this.sanitizeForLog(ctx.messages ?? []) as unknown[],
			};
		}

		if (ctx.system !== undefined) {
			return {
				system: typeof ctx.system === "string" ? ctx.system : JSON.stringify(ctx.system),
				messages: this.sanitizeForLog(ctx.messages ?? []) as unknown[],
			};
		}

		// Check for messages array with potential system messages
		const messages = Array.isArray(ctx.messages) ? ctx.messages : [];
		const systemMessages = messages
			.filter((m: unknown) => (m as Record<string, unknown>)?.role === "system")
			.map((m: unknown) => (m as Record<string, unknown>).content);

		return {
			system: systemMessages.length > 0 ? String(systemMessages[0]) : undefined,
			messages: this.sanitizeForLog(
				messages.filter((m: unknown) => (m as Record<string, unknown>)?.role !== "system"),
			) as unknown[],
		};
	}

	private extractResponseContent(response: unknown): {
		role: string;
		content: unknown;
		usage?: unknown;
		stopReason?: string;
		id?: string;
		model?: string;
	} {
		if (!response || typeof response !== "object") {
			return { role: "unknown", content: null };
		}

		const resp = response as Record<string, unknown>;

		return {
			id: resp.id as string | undefined,
			model: resp.model as string | undefined,
			role: (resp.role as string) || "assistant",
			content: this.sanitizeForLog(resp.content ?? resp.choices ?? resp.message ?? resp),
			usage: this.sanitizeForLog(resp.usage ?? resp.usageMetadata ?? resp.llmUsage),
			stopReason:
				(resp.stopReason as string) ??
				((Array.isArray(resp.choices) && resp.choices[0]
					? (resp.choices[0] as Record<string, unknown>)?.finishReason
					: undefined) as string | undefined),
		};
	}

	logRequestStart(provider: string, modelId: string, context: unknown, options?: unknown): string {
		this.ensureInitialized();

		const callId = this.generateCallId();

		if (!this.config.enabled) return callId;

		const { system, messages } = this.extractSystemPrompt(context);

		const entry: LlmLogEntry = {
			timestamp: new Date().toISOString(),
			callId,
			provider,
			modelId,
			request: {
				system,
				messages,
				options: this.sanitizeForLog(options),
			},
			durationMs: 0,
		};

		this.writeLog(callId, entry);
		return callId;
	}

	logRequestEnd(
		callId: string,
		provider: string,
		modelId: string,
		context: unknown,
		options: unknown | undefined,
		response: unknown,
		durationMs: number,
		error?: Error,
	): void {
		this.ensureInitialized();

		if (!this.config.enabled) return;

		const { system, messages } = this.extractSystemPrompt(context);

		const entry: LlmLogEntry = {
			timestamp: new Date().toISOString(),
			callId,
			provider,
			modelId,
			request: {
				system,
				messages,
				options: this.sanitizeForLog(options),
			},
			response: error
				? { role: "error", content: error.message, error: error.stack }
				: this.extractResponseContent(response),
			durationMs,
		};

		this.writeLog(callId, entry);
	}

	private writeLog(_callId: string, entry: LlmLogEntry): void {
		try {
			const filename = `llm_${new Date().toISOString().slice(0, 10)}.jsonl`;
			const filepath = join(this.config.dir, filename);
			const line = `${JSON.stringify(entry)}\n`;
			appendFileSync(filepath, line, "utf8");
		} catch (err) {
			console.error("[LlmLogger] Failed to write log:", err);
		}
	}

	/**
	 * Wrap an AssistantMessageEventStream to log input/output.
	 * Returns the original stream with logging side effects.
	 */
	wrapStream(
		stream: AssistantMessageEventStream,
		callId: string,
		provider: string,
		modelId: string,
		context: unknown,
		options: unknown | undefined,
	): AssistantMessageEventStream {
		this.ensureInitialized();

		if (!this.config.enabled) return stream;

		const startTime = Date.now();

		// Listen for completion to log the result
		stream.result().then(
			(result) => {
				const durationMs = Date.now() - startTime;
				if (result instanceof Error) {
					this.logRequestEnd(callId, provider, modelId, context, options, undefined, durationMs, result);
				} else {
					this.logRequestEnd(callId, provider, modelId, context, options, result, durationMs);
				}
			},
			(err) => {
				const durationMs = Date.now() - startTime;
				const errorObj = err instanceof Error ? err : new Error(String(err));
				this.logRequestEnd(callId, provider, modelId, context, options, undefined, durationMs, errorObj);
			},
		);

		return stream;
	}

	/**
	 * Write a marker entry to the log file (useful for debugging session boundaries).
	 */
	logMarker(label: string, data?: unknown): void {
		this.ensureInitialized();

		if (!this.config.enabled) return;

		const entry: LlmLogEntry = {
			timestamp: new Date().toISOString(),
			callId: `marker_${Date.now()}`,
			provider: "system",
			modelId: "system",
			request: {
				messages: [{ role: "system", content: label }],
				options: data,
			},
			durationMs: 0,
		};

		this.writeLog(entry.callId, entry);
	}
}

// Singleton instance
export const llmLogger = new LlmLogger();
