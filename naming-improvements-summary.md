# Naming Improvements for Trace Snapshot Functionality

## Context

This document summarizes improvements made to the naming conventions in the Generative AI Toolkit's tracing system, specifically focusing on the functionality that handles in-progress traces.

## Problem Statement

The original codebase used terminology that wasn't optimally descriptive:

- `PreviewableTracer` - A Protocol for tracers that can handle in-progress traces
- `preview()` - A method to process in-progress traces
- `share_preview()` - A method on Trace to share its current state
- `preview_enabled` - A flag indicating whether the preview functionality is enabled

These names didn't clearly communicate:
1. The in-progress nature of the traces being handled
2. The relationship with the existing persistence mechanism
3. The point-in-time aspect of capturing trace state

## Naming Exploration

We explored several alternative naming schemes focusing on different conceptual frameworks:

1. **"In-Progress" focus**:
   - `InProgressTracer`, `process_in_progress()`, `share_in_progress_state()`

2. **"Partial" focus**:
   - `PartialTraceHandler`, `handle_partial()`, `emit_partial_state()`
   
3. **"Snapshot" focus**:
   - `TraceSnapshotHandler`, `handle_snapshot()`, `emit_snapshot()`

4. **"Evolving" focus**:
   - `EvolvingTraceHandler`, `process_evolving()`, `share_evolving_state()`

After careful consideration, we chose the **"Snapshot"** terminology as it best communicated the point-in-time nature of capturing the current state of a trace that is still evolving.

## Changes Made

We implemented a consistent naming scheme around the "snapshot" concept:

### Protocol (Interface):
- ✓ `PreviewableTracer` → `SnapshotCapableTracer`

### Attributes:
- ✓ `preview_enabled` → `snapshot_enabled`

### Methods:
- ✓ `preview(trace)` → `persist_snapshot(trace)`
- ✓ `share_preview()` → `emit_snapshot()`

### Parameters and Fields:
- ✓ `share_preview` parameter → `snapshot_handler` 
- ✓ `self._share_preview` → `self._snapshot_handler`
- ✓ `previewer` → `snapshot_handler`

### Files Modified:
- `src/generative_ai_toolkit/tracer/tracer.py` 
- `src/generative_ai_toolkit/tracer/trace.py`
- `src/generative_ai_toolkit/tracer/__init__.py`
- `src/generative_ai_toolkit/agent/agent.py`

## Rationale

The new naming scheme:

1. **Creates Clearer Mental Model**: The "snapshot" term communicates that we're capturing the trace at a specific point in time.

2. **Establishes Consistent Patterns**: Using `persist_snapshot()` alongside the existing `persist()` creates a clear relationship between handling complete traces and in-progress trace snapshots.

3. **Self-Documents the Code**: The names better express the intent and purpose of these components, reducing the need for explanatory comments.

4. **Uses Familiar Terminology**: "Snapshot" is a widely understood concept in software development, making the code more approachable.
## Implementation Update

The initial implementation updated the core tracing components but missed some references in the agent implementation. This has now been addressed by:

1. **Updated Agent Method Calls**: All occurrences of `share_preview()` in the `agent.py` file have been changed to `emit_snapshot()`, ensuring consistency throughout the codebase.

2. **Completed End-to-End Implementation**: The naming scheme is now consistent across all components that interact with trace snapshots.

## Conclusion

These naming improvements enhance the readability and maintainability of the code by making the intent clearer. The relationship between components is now more apparent, and new developers will more easily understand the purpose of these classes and methods without diving deep into the implementation details.

By focusing on semantic accuracy and consistency, these changes follow good software engineering practices where names should communicate meaning and purpose.