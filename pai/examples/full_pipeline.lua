local Execution = require("execution")

local config = {
	input_data = pbridge.create_dataframe({
		{ id = 1, income = 5000, debt = 2000, credit_score = 680 },
		{ id = 2, income = 6000, debt = 2500, credit_score = 720 },
		{ id = 3, income = 4000, debt = 4500, credit_score = 580 },
	}),

	expected_columns = { "id", "income", "debt", "credit_score" },

	external_apis = nil,

	transformations = {
		{
			type = "constraints",
			constraints = {
				adult = "id >= 1",
				good_income = "income > debt * 0.4",
				credit_ok = "credit_score >= 620",
			},
		},
		{
			type = "streaming",
			value_col = "income",
			window = 2,
		},
	},

	batch_size = 1,
	batch_processing_function = "no_op",
}

local result = Execution.execute_data_pipeline(config)

print("Pipeline success:", result.success)
print("Output rows:", #result.output_data)
