# Copilot Instructions for RapidSwot

This document provides instructions for AI coding agents to effectively contribute to the RapidSwot codebase.

## Project Overview

RapidSwot is an AI-powered browser automation agent. It uses a sophisticated agent-based architecture to interpret high-level user intentions and translate them into concrete actions, such as browsing websites, extracting information, and interacting with web elements.

The project is divided into three main parts:

*   **Backend**: A FastAPI application that serves the main API.
*   **Frontend**: A Vue.js application that provides the user interface.
*   **Core**: The core agent logic, which is written in Python and uses the `google.adk` framework.

## Developer Workflows

### Backend

The backend is a FastAPI application located in the `backend` directory. To run the backend in development mode, use the following command:

```bash
./dev.sh
```

This will start the FastAPI server on port 8000, along with MongoDB and Redis.

### Frontend

The frontend is a Vue.js application located in the `frontend` directory. To run the frontend in development mode, use the following commands:

```bash
cd frontend
npm install
npm run dev
```

This will start the Vite development server.

## Architecture

The core of the application is the `RapidSwotAgent`, which is defined in `src/agent/core/agent.py`. This agent is responsible for orchestrating the entire process of interpreting user intentions and executing them.

The agent's architecture is based on the following components:

*   **`IntentProcessor`**: This class uses an LLM to convert a high-level user intent into a structured `ActionPlan`.
*   **`ExecutionEngine`**: This class executes the `ActionPlan` by calling the appropriate tools.
*   **`RapidSwotAgent`**: This is the main agent class that orchestrates the entire process.

### Tools

The agent has a set of tools that it can use to perform various actions. These tools are located in the `src/agent` directory. The most important tool is the `BrowserController`, which is defined in `src/agent/browser.py`. This tool uses Playwright to automate browser actions.

## Key Files

*   `dev.sh`: The main development script.
*   `backend/app/main.py`: The entry point for the FastAPI application.
*   `backend/app/interfaces/api/routes.py`: The API routes for the backend.
*   `frontend/package.json`: The frontend's dependencies and scripts.
*   `src/agent/core/agent.py`: The core agent logic.
*   `src/agent/browser.py`: The browser automation tool.

## Project-Specific Conventions

*   The backend and frontend are in separate directories, `backend` and `frontend`, respectively.
*   The core agent logic is in the `src` directory.
*   The agent is built on top of the `google.adk` framework.
*   The backend uses FastAPI, MongoDB, and Redis.
*   The frontend uses Vue.js and Vite.
