import { parseArgs } from "node:util";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { resolveDirs } from "./config.js";
import { createServer } from "./server.js";

const { values } = parseArgs({ options: { data: { type: "string" }, config: { type: "string" } }, strict: true });
const server = createServer(resolveDirs(values, process.env));
await server.connect(new StdioServerTransport());
