import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createServer } from "./server.js";

const server = createServer(process.env["VIDEO_BENCH_DATA_DIR"] ?? "./data");
await server.connect(new StdioServerTransport());
