-- Frank-Wolfe algorithm
local eps = torch.pow(2, -52)
helpers = require 'helpers.helperFn'

local function FW(options)

	print("Starting Frank-Wolfe \n")
	local featureSize = options.featureSize
	local featureMapSize = options.nclasses * featureSize
	local n = options.trainset_size 
	local debug_flag = options.debug_flag
	local w = torch.Tensor(featureMapSize):zero()
	local l = 0
	local K = options.num_passes 
	local lambda = options.lambda
	local ws = torch.Tensor(featureMapSize):zero()
	local ls = 0
	local epsilon = options.epsilon 
	local primalArr = torch.Tensor(K+1):zero()
	local dualArr = torch.Tensor(K+1):zero()
	local timeArr = torch.Tensor(K+1):zero()
	local numIter = 1
	local gamma = 0

	print("Using " .. n .. " training examples")

	for k = 0, K do
		ws = ws:zero()
		ls = 0

		for i = 1, n do
			local x = options.minitrainset_data[i]
			local y = options.minitrainset_label[i]
			local max_y = helpers.maxOracle(x, y, w)
			local psi_i = helpers.featureVec(x, y) - helpers.featureVec(x, max_y)
			ws = ws + psi_i 
			ls = ls + helpers.loss(y, max_y)
			assert(helpers.loss(y, max_y) - w:dot(psi_i) >= -1e-12)
		end

		ws = 1/(n*lambda) * ws
		ls = 1/n * ls

		if options.do_line_search then
			gamma = (w:dot(w - ws) - 1/lambda * (l - ls))/((w - ws):dot(w - ws) + eps)
			gamma = math.max(0, math.min(1, gamma))	
		else
			gamma = 2/(k + 2)
		end
	
		w = (1 - gamma)*w + gamma * ws
		l = (1 - gamma)*l + gamma * ls

		if options.debug == 1 then
			local f = helpers.getDual(w, l, lambda)
			local truedualGap = helpers.duality_gap(w, lambda, l, n, options.minitrainset_data, options.minitrainset_label) 
			local primal = f + truedualGap

			assert(truedualGap > 0, 'truedualGap: ' .. truedualGap .. '\n')
			assert(primal > truedualGap, 'primal: ' .. primal .. '  truedualGap: ' .. truedualGap .. '\n')
			assert(primal > f, 'Primal: ' .. primal .. '  Dual: ' .. (f) .. '\n')
	
			local training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, w)
			io.write(string.format("pass %d, SVM primal = %.6f, SVM dual = %.6f, duality gap = %.6f, training error = %.6f\n", k, primal, f, truedualGap, training_loss))

			primalArr[numIter] = primal
			dualArr[numIter] = f
			timeArr[numIter] = timer:time().real
			numIter = numIter + 1
			gnuplot.pdffigure('FW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
			gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
			gnuplot.xlabel('Time(s)')
			gnuplot.ylabel('Value')
			gnuplot.plotflush()

			if truedualGap < epsilon then
				print("Duality gap below threshold \n")
				break
			end

		end

	end

	print(string.format("elapsed time: %.2f sec\n", timer:time().real))

	local training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, w)

	print("Training loss: " .. training_loss .. "\n")
	t = 0
	iterArr = torch.Tensor(numIter)
	iterArr:apply(function()
		t = t + 1
		return t
	end)
	gnuplot.pdffigure('FW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
	gnuplot.plot(
--	{'Primal', iterArr, primalArr[{{1, numIter - 1}}]},
	{'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'})
	gnuplot.xlabel('Time (s)')
	gnuplot.ylabel('Value')
	gnuplot.plotflush()
	return timeArr, dualArr, numIter
end

return{FW = FW}
