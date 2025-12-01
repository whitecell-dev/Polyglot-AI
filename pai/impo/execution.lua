-- execution.lua
-- Execution orchestration and bridge coordination

local pbridge = require("pbridge")
local workflow = require("workflow")
local apiretry = require("api_retry")

local Execution = {}

-- CANONICAL EXECUTION #1: Full Data Processing Pipeline
function Execution.execute_data_pipeline(config)
	print("[IMPO.Execution] INFO: Starting data processing pipeline")

	local tracker = workflow.create_workflow_tracker("pipeline_started")

	-- Step 1: Validate input structure
	tracker = workflow.workflow_transition(tracker, "validation", "Validating input data")

	local validation = workflow.validate_dataframe_structure(config.input_data, config.expected_columns)

	if not validation.structural_valid then
		print("[IMPO.Execution] ERROR: Input validation failed")
		tracker = workflow.workflow_finalize(tracker, "validation_failed")
		return {
			success = false,
			error = "validation_failed",
			validation_result = validation,
			tracker = tracker,
		}
	end

	-- Step 2: External data enrichment (pure IMPO - I/O only)
	tracker = workflow.workflow_transition(tracker, "enrichment", "Enriching data with external APIs")

	local enriched_data = config.input_data
	if config.external_apis then
		for _, api_config in ipairs(config.external_apis) do
			local api_result = apiretry.external_api_call(
				api_config.name,
				api_config.max_retries or 3,
				api_config.base_backoff_ms or 100
			)

			if api_result.success then
				-- Bridge call to ALBEO for data enrichment
				enriched_data = pbridge.enrich_data(enriched_data, api_config.enrichment_function, api_result)
			else
				print("[IMPO.Execution] WARN: API call failed: " .. api_config.name)
			end
		end
	end

	-- Step 3: Execute ALBEO transformations (pure computation)
	tracker = workflow.workflow_transition(tracker, "transformation", "Executing algebraic transformations")

	local transformed_data = enriched_data
	for _, transform in ipairs(config.transformations) do
		-- Bridge call to appropriate ALBEO function
		if transform.type == "lenses" then
			transformed_data = pbridge.json_lens_extract(transformed_data, transform.column, transform.path)
		elseif transform.type == "streaming" then
			transformed_data = pbridge.streaming_rolling(transformed_data, transform.value_col, transform.window)
		elseif transform.type == "graph" then
			transformed_data = pbridge.graph_neighbors(transformed_data, transform.src_col, transform.dst_col)
		elseif transform.type == "temporal" then
			transformed_data =
				pbridge.temporal_lag_lead(transformed_data, transform.value_col, transform.lags, transform.leads)
		elseif transform.type == "constraints" then
			transformed_data = pbridge.constraint_solve(transformed_data, transform.constraints)
		end

		print("[IMPO.Execution] INFO: Applied transformation: " .. transform.type)
	end

	-- Step 4: Batch processing if needed
	tracker = workflow.workflow_transition(tracker, "batch_processing", "Processing data in batches")

	local final_result = transformed_data
	if config.batch_size and config.batch_size > 0 then
		local batch_result = workflow.batch_process(
			{ transformed_data }, -- In real scenario, split into batches
			function(batch)
				-- Bridge call to ALBEO for batch processing
				return pbridge.batch_process(batch, config.batch_processing_function)
			end,
			config.batch_size
		)
		final_result = batch_result
	end

	-- Step 5: Output preparation
	tracker = workflow.workflow_transition(tracker, "output_preparation", "Preparing final output")

	-- Convert to primitives for cross-language transmission
	local output_primitives = pbridge.to_primitives(final_result)

	-- Finalize
	tracker = workflow.workflow_finalize(tracker, "pipeline_completed")

	return {
		success = true,
		output_data = output_primitives,
		tracker = tracker,
		validation = validation,
		transformations_applied = #config.transformations,
	}
end

-- CANONICAL EXECUTION #2: Real-time Stream Processing
function Execution.execute_stream_processing(stream_config)
	print("[IMPO.Execution] INFO: Starting stream processing")

	local tracker = workflow.create_workflow_tracker("stream_started")
	local window_buffer = {}
	local results = {}

	-- Simulate stream processing loop
	for i = 1, stream_config.max_iterations or 100 do
		tracker = workflow.workflow_transition(tracker, "stream_window_" .. i, "Processing stream window " .. i)

		-- Simulate receiving data point (in production: from message queue, socket, etc.)
		local data_point = {
			timestamp = os.time(),
			value = math.random() * 100,
			id = "point_" .. i,
		}

		table.insert(window_buffer, data_point)

		-- Process window when buffer reaches size
		if #window_buffer >= stream_config.window_size then
			-- Convert buffer to DataFrame format for ALBEO
			local window_df = pbridge.create_dataframe(window_buffer)

			-- Bridge call to ALBEO for stream processing
			local window_result = pbridge.streaming_rolling(window_df, "value", stream_config.window_size)

			-- Convert result to primitives
			local primitive_result = pbridge.to_primitives(window_result)
			table.insert(results, {
				window_index = i,
				result = primitive_result,
				timestamp = os.time(),
			})

			-- Clear buffer (sliding window)
			if stream_config.window_type == "sliding" then
				table.remove(window_buffer, 1) -- Remove oldest
			else -- tumbling window
				window_buffer = {}
			end

			print(string.format("[IMPO.Execution] INFO: Processed window %d, %d results so far", i, #results))
		end

		-- Rate limiting for simulation
		if stream_config.simulated_delay then
			apiretry._sleep(stream_config.simulated_delay)
		end
	end

	tracker = workflow.workflow_finalize(tracker, "stream_completed")

	return {
		success = true,
		results = results,
		tracker = tracker,
		total_windows_processed = #results,
		final_buffer_size = #window_buffer,
	}
end

-- CANONICAL EXECUTION #3: Error Recovery Workflow
function Execution.execute_with_error_recovery(execution_function, recovery_config)
	print("[IMPO.Execution] INFO: Starting execution with error recovery")

	local max_attempts = recovery_config.max_attempts or 3
	local attempt = 1
	local last_error = nil

	while attempt <= max_attempts do
		print(string.format("[IMPO.Execution] INFO: Attempt %d/%d", attempt, max_attempts))

		local success, result = pcall(execution_function)

		if success then
			print("[IMPO.Execution] INFO: Execution succeeded on attempt " .. attempt)
			return {
				success = true,
				result = result,
				attempts = attempt,
				recovered = attempt > 1,
			}
		else
			last_error = result
			print(string.format("[IMPO.Execution] WARN: Attempt %d failed: %s", attempt, tostring(result)))

			-- Apply recovery strategy
			if recovery_config.strategy == "exponential_backoff" then
				local backoff_ms = recovery_config.base_backoff_ms or 100
				backoff_ms = backoff_ms * (2 ^ (attempt - 1))
				print(string.format("[IMPO.Execution] INFO: Backing off for %dms", backoff_ms))
				apiretry._sleep(backoff_ms / 1000)
			elseif recovery_config.strategy == "fallback_function" then
				print("[IMPO.Execution] INFO: Trying fallback function")
				execution_function = recovery_config.fallback_function
			elseif recovery_config.strategy == "degraded_mode" then
				print("[IMPO.Execution] INFO: Switching to degraded mode")
				-- Adjust execution parameters
				if recovery_config.degraded_params then
					for k, v in pairs(recovery_config.degraded_params) do
						execution_function[k] = v
					end
				end
			end

			attempt = attempt + 1
		end
	end

	print("[IMPO.Execution] ERROR: All recovery attempts failed")
	return {
		success = false,
		error = last_error,
		attempts = max_attempts,
		recovered = false,
	}
end

return Execution
