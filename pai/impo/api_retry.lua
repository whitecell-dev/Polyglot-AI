-- api_retry.lua
-- External API calls with retry logic and backoff - pure IMPO layer

local ApiRetry = {}

-- CANONICAL API RETRY #1: Exponential Backoff Retry
function ApiRetry.external_api_call(api_name, max_retries, base_backoff_ms)
	print("[IMPO.API] INFO: Starting " .. api_name)

	local result = {
		success = false,
		attempts = 0,
		backoff_sequence = {},
		error_messages = {},
	}

	local backoff = base_backoff_ms

	while result.attempts < max_retries do
		result.attempts = result.attempts + 1

		print(string.format("[IMPO.API] INFO: Attempt %d/%d for %s", result.attempts, max_retries, api_name))

		-- Simulate API call (in production: actual HTTP/RPC via ngx.location.capture or socket)
		local success, error_msg = ApiRetry._simulate_api_call(api_name)

		if success then
			result.success = true
			result.final_backoff = backoff
			print(string.format("[IMPO.API] INFO: %s succeeded after %d attempts", api_name, result.attempts))
			return result
		else
			table.insert(result.backoff_sequence, backoff)
			table.insert(result.error_messages, error_msg)

			print(
				string.format(
					"[IMPO.API] WARN: Attempt %d failed: %s, backoff=%dms",
					result.attempts,
					error_msg,
					backoff
				)
			)

			-- Exponential backoff with jitter
			local jitter = math.random() * 0.3 + 0.85 -- 85-115% of backoff
			backoff = math.floor(backoff * 2 * jitter)

			-- In production: use ngx.sleep() or socket.sleep()
			ApiRetry._sleep(backoff / 1000) -- Convert ms to seconds
		end
	end

	print("[IMPO.API] ERROR: " .. api_name .. " exhausted all retries after " .. result.attempts .. " attempts")
	return result
end

-- CANONICAL API RETRY #2: Circuit Breaker Pattern
function ApiRetry.circuit_breaker_call(api_name, circuit_breaker)
	print("[IMPO.API] INFO: Circuit breaker call to " .. api_name)

	-- Check circuit breaker state
	if circuit_breaker.state == "open" then
		local now = os.time()
		if now - circuit_breaker.last_failure_time < circuit_breaker.reset_timeout then
			print("[IMPO.API] WARN: Circuit breaker is open, failing fast")
			return {
				success = false,
				error = "circuit_breaker_open",
				message = "Service unavailable due to circuit breaker",
			}
		else
			-- Try to reset circuit breaker (half-open state)
			circuit_breaker.state = "half_open"
			print("[IMPO.API] INFO: Circuit breaker reset to half-open")
		end
	end

	-- Make the API call
	local success, error_msg = ApiRetry._simulate_api_call(api_name)

	-- Update circuit breaker
	if success then
		circuit_breaker.failure_count = 0
		circuit_breaker.state = "closed"
		print("[IMPO.API] INFO: Circuit breaker closed, call succeeded")
	else
		circuit_breaker.failure_count = circuit_breaker.failure_count + 1
		circuit_breaker.last_failure_time = os.time()

		if circuit_breaker.failure_count >= circuit_breaker.failure_threshold then
			circuit_breaker.state = "open"
			print("[IMPO.API] ERROR: Circuit breaker tripped to open")
		else
			print(
				string.format(
					"[IMPO.API] WARN: Circuit breaker failure count: %d/%d",
					circuit_breaker.failure_count,
					circuit_breaker.failure_threshold
				)
			)
		end
	end

	return {
		success = success,
		error = error_msg,
		circuit_breaker_state = circuit_breaker.state,
		failure_count = circuit_breaker.failure_count,
	}
end

-- CANONICAL API RETRY #3: Batch API Calls with Rate Limiting
function ApiRetry.batch_api_calls(api_calls, max_concurrent, rate_limit_per_second)
	print(string.format("[IMPO.API] INFO: Starting batch of %d API calls", #api_calls))

	local results = {}
	local completed = 0
	local failed = 0
	local start_time = os.time()

	-- Rate limiting window
	local rate_window_start = os.time()
	local calls_in_window = 0

	for i = 1, #api_calls do
		local api_call = api_calls[i]

		-- Rate limiting check
		local now = os.time()
		if now - rate_window_start >= 1 then -- New second, reset window
			rate_window_start = now
			calls_in_window = 0
		end

		if calls_in_window >= rate_limit_per_second then
			-- Wait for next second
			ApiRetry._sleep(1)
			rate_window_start = os.time()
			calls_in_window = 0
		end

		calls_in_window = calls_in_window + 1

		-- Make API call with retry
		local result =
			ApiRetry.external_api_call(api_call.name, api_call.max_retries or 3, api_call.base_backoff_ms or 100)

		result.call_name = api_call.name
		result.call_index = i

		table.insert(results, result)

		if result.success then
			completed = completed + 1
		else
			failed = failed + 1
		end

		-- Progress logging
		if i % math.ceil(#api_calls / 10) == 0 then
			print(string.format("[IMPO.API] INFO: Progress: %d/%d calls processed", i, #api_calls))
		end
	end

	local total_time = os.time() - start_time

	return {
		results = results,
		summary = {
			total_calls = #api_calls,
			completed = completed,
			failed = failed,
			success_rate = completed / #api_calls,
			total_time_seconds = total_time,
			calls_per_second = #api_calls / total_time,
		},
	}
end

-- Internal helper function (simulates API call)
function ApiRetry._simulate_api_call(api_name)
	-- Simulate API success/failure
	-- In production: replace with actual HTTP call

	-- 70% success rate for simulation
	if math.random() < 0.7 then
		return true, nil
	else
		local errors = {
			"network_timeout",
			"server_error_500",
			"rate_limit_exceeded",
			"connection_refused",
		}
		local error_msg = errors[math.random(1, #errors)]
		return false, error_msg
	end
end

-- Internal helper function (sleep)
function ApiRetry._sleep(seconds)
	-- In OpenResty: use ngx.sleep(seconds)
	-- In standalone Lua: use socket.sleep(seconds) from luasocket
	-- For now, simulate sleep
	local start = os.time()
	while os.time() - start < seconds do
		-- Busy wait (in production use proper sleep)
	end
end

return ApiRetry
