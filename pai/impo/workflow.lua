-- workflow.lua
-- Workflow orchestration and state machine management for pAI IMPO layer

local pbridge = require("pbridge") -- Bridge to Python ALBEO functions

local Workflow = {}

-- CANONICAL WORKFLOW #1: Structure Validation
-- This is the IMPO layer's responsibility - validation before passing to ALBEO
function Workflow.validate_dataframe_structure(df, expected_columns)
	print("[IMPO.Workflow] INFO: Validating DataFrame structure")

	-- SYNTAX ENFORCEMENT: Only bridge calls, no direct DataFrame access
	local shape = pbridge.shape_info(df)
	local schema = pbridge.schema_validation(df, expected_columns)

	-- Business logic based on primitives (IMPO's job)
	if schema.is_valid and shape.rows > 0 then
		print(string.format("[IMPO.Workflow] INFO: Structure valid: %dx%d", shape.rows, shape.cols))
		return {
			structural_valid = true,
			shape_info = shape,
			schema_info = schema,
		}
	else
		print("[IMPO.Workflow] ERROR: Structure validation failed")
		return {
			structural_valid = false,
			shape_info = shape,
			schema_info = schema,
		}
	end
end

-- CANONICAL WORKFLOW #2: State Machine Tracker
function Workflow.create_workflow_tracker(initial_state)
	print("[IMPO.Workflow] INFO: Created tracker: " .. initial_state)
	return {
		current_state = initial_state,
		state_history = { initial_state },
		transitions = 0,
		start_time = os.time(),
	}
end

function Workflow.workflow_transition(tracker, new_state, reason)
	tracker.transitions = tracker.transitions + 1
	table.insert(tracker.state_history, new_state)
	tracker.current_state = new_state

	local prev_state = tracker.state_history[#tracker.state_history - 1]
	print(string.format("[IMPO.Workflow] INFO: State %s -> %s: %s", prev_state, new_state, reason))

	return tracker
end

function Workflow.workflow_finalize(tracker, final_status)
	tracker.duration_seconds = os.time() - tracker.start_time
	tracker.final_status = final_status

	print(
		string.format(
			"[IMPO.Workflow] INFO: Workflow finalized: %d transitions, %ds duration, status=%s",
			tracker.transitions,
			tracker.duration_seconds,
			final_status
		)
	)

	return tracker
end

-- CANONICAL WORKFLOW #3: Batch Processing Orchestration
function Workflow.batch_process(dataframes, processing_function, batch_size)
	print(string.format("[IMPO.Workflow] INFO: Starting batch processing of %d dataframes", #dataframes))

	local tracker = Workflow.create_workflow_tracker("batch_processing")
	local results = {}

	for i = 1, #dataframes, batch_size do
		local batch_end = math.min(i + batch_size - 1, #dataframes)
		local batch = {}

		-- Collect batch
		for j = i, batch_end do
			table.insert(batch, dataframes[j])
		end

		-- Transition state
		tracker = Workflow.workflow_transition(
			tracker,
			"processing_batch",
			string.format("Processing batch %d-%d of %d", i, batch_end, #dataframes)
		)

		-- Process batch (orchestration only - computation happens in ALBEO)
		local batch_result = processing_function(batch)
		table.insert(results, batch_result)
	end

	tracker = Workflow.workflow_finalize(tracker, "completed")

	return {
		results = results,
		metadata = tracker,
		total_batches = math.ceil(#dataframes / batch_size),
		total_dataframes = #dataframes,
	}
end

-- CANONICAL WORKFLOW #4: Data Pipeline Orchestration
function Workflow.execute_data_pipeline(df, pipeline_steps)
	print("[IMPO.Workflow] INFO: Executing data pipeline with " .. #pipeline_steps .. " steps")

	local tracker = Workflow.create_workflow_tracker("pipeline_started")
	local current_df = df

	for i, step in ipairs(pipeline_steps) do
		tracker = Workflow.workflow_transition(tracker, "step_" .. i, "Executing pipeline step: " .. step.name)

		-- Each step is a bridge call to ALBEO
		-- IMPO orchestrates, ALBEO computes
		if step.type == "validation" then
			local validation = Workflow.validate_dataframe_structure(current_df, step.expected_columns)
			if not validation.structural_valid then
				error("Pipeline validation failed at step " .. i)
			end
		elseif step.type == "transformation" then
			-- Bridge call to ALBEO transformation
			current_df = pbridge.execute_transformation(current_df, step.function_name, step.params)
		elseif step.type == "filtering" then
			-- Bridge call to ALBEO filtering
			current_df = pbridge.execute_filtering(current_df, step.condition)
		end

		print(string.format("[IMPO.Workflow] INFO: Step %d completed successfully", i))
	end

	tracker = Workflow.workflow_finalize(tracker, "pipeline_completed")

	return {
		result = current_df,
		pipeline_metadata = tracker,
		steps_completed = #pipeline_steps,
	}
end

return Workflow
