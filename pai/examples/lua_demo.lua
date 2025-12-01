local pbridge = require("pbridge")

local df = pbridge.create_dataframe({
	{ income = 5000, debt = 2000 },
	{ income = 6000, debt = 2500 },
})

local out = pbridge.calculate_ratios({ df = df })
print(out[2].dti_ratio)
