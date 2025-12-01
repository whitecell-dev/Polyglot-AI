-- pai/impo/pbridge.lua
local http = require("socket.http")
local json = require("dkjson")

local pbridge = {}

local BASE_URL = "http://127.0.0.1:7777/rpc"

-- Low-level RPC function
local function rpc_call(method, params)
	local body = json.encode({
		method = method,
		params = params,
	})

	local response_body = {}
	local _, code = http.request({
		url = BASE_URL,
		method = "POST",
		source = ltn12.source.string(body),
		headers = {
			["Content-Type"] = "application/json",
			["Content-Length"] = tostring(#body),
		},
		sink = ltn12.sink.table(response_body),
	})

	local raw = table.concat(response_body)
	local decoded = json.decode(raw)

	if decoded.has_error then
		error(decoded.error or "Unknown Python error")
	end

	return decoded.result
end

-- Helper: turn Lua table → Python DataFrame (list of dicts)
function pbridge.create_dataframe(rows)
	return rows
end

-- AUTO-GENERATED BRIDGE FUNCTIONS
local function make(method)
	return function(...)
		local args = { ... }
		local params = {}

		-- Convert positional args → named params for Python
		-- (Python server expects kwargs)
		if #args == 1 then
			params = args[1]
		end
		if #args == 2 then
			params = { df = args[1], column = args[2] }
		end
		if #args == 3 then
			params = { df = args[1], column = args[2], path = args[3] }
		end

		return rpc_call(method, params)
	end
end

-- Structural Bridge Functions
pbridge.shape_info = make("shape_info")
pbridge.schema_validation = make("schema_validation")
pbridge.basic_info = make("basic_info")
pbridge.get_column_values = make("get_column_values")
pbridge.json_lens_extract = make("json_lens_extract")
pbridge.extract_numerical_stats = make("extract_numerical_stats")
pbridge.describe_dataframe = make("describe_dataframe")

-- ALBEO functions
pbridge.calculate_ratios = make("calculate_ratios")
pbridge.calculate_aggregate_metrics = make("calculate_aggregate_metrics")
pbridge.ml_feature_engineering = make("ml_feature_engineering")

pbridge.constraint_solve = make("constraint_solve")

pbridge.temporal_lag_lead = make("temporal_lag_lead")
pbridge.temporal_resample = make("temporal_resample")
pbridge.temporal_join = make("temporal_join")

pbridge.streaming_rolling = make("streaming_rolling")
pbridge.streaming_expanding = make("streaming_expanding")
pbridge.streaming_duckdb_window = make("streaming_duckdb_window")

pbridge.graph_neighbors = make("graph_neighbors")
pbridge.graph_depth_limited_paths = make("graph_depth_limited_paths")

return pbridge
