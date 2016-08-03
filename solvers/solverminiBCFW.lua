-- Minibatch BCFW algorithm
local eps = torch.pow(2, -52)
helpers = require 'helpers.helperFn'

function minibatchBCFW(options, minibatchSize)

	print("Starting Minibatch Block-Coordinate Frank-Wolfe \n")
	local featureSize = options.featureSize
	local featureMapSize = options.nclasses * featureSize
	local n = options.trainset_size 
	local w = torch.Tensor(featureMapSize):zero()
	local l = 0
	local minibatchNum = n/minibatchSize 	-- total number of minibatches
	local w_i = torch.Tensor(minibatchNum, featureMapSize):zero() -- i indexes minibatch
	local l_i = torch.Tensor(minibatchNum):zero()
	local ws = torch.Tensor(featureMapSize):zero()
	local ls = 0
	local K = options.num_passes
	local lambda = options.lambda 
	local epsilon = options.epsilon 
	local primalArr = torch.Tensor(K+1):zero()
	local dualArr = torch.Tensor(K+1):zero()
	local timeArr = torch.Tensor(K+1):zero()
	local numIter = 1
	local wAvg = torch.Tensor(featureMapSize):zero()
	local lAvg = 0 

	print("Using " .. n .. " training examples")	

	local k = 0
	for p = 1, K do  -- indexes the number of passes through data
		local i = 0
		for dummy = 1, minibatchNum do  -- each iteration is gradient and weight updated using one minibatch
			i = dummy
			ws = ws:zero()
			ls = 0

			for j = 1, minibatchSize do
				local x = options.minitrainset_data[(i - 1)*minibatchSize + j]
				local y = options.minitrainset_label[(i - 1)*minibatchSize + j]
				local max_y = helpers.maxOracle(x, y, w)
				local psi_i = helpers.featureVec(x, y) - helpers.featureVec(x, max_y)
				ws = ws + psi_i 
				ls = ls + helpers.loss(y, max_y)
				assert(helpers.loss(y, max_y) - w:dot(psi_i) >= -1e-12)
			end

			ws = 1/(minibatchSize*lambda) * ws
			ls = 1/minibatchSize * ls
			local gamma = 0

			if options.do_line_search == 1 then
--				print("Doing line search")
				gamma = (w:dot(w_i[i] - ws) - 1/lambda * (l_i[i] - ls))/((w_i[i] - ws):dot(w_i[i] - ws) + eps)
				if gamma < 0 then
					gamma = 0
				elseif gamma > 1 then
					gamma = 1
				end
			else
--				print("Not doing line search")
				gamma = 2*n/(k + 2*n)	
			end

			local new_w_i = (1 - gamma)*w_i[i] + gamma * ws
			local new_l_i = (1 - gamma)*l_i[i] + gamma * ls

			w = w + new_w_i - w_i[i]
			l = l + new_l_i - l_i[i]
		
			w_i[i] = new_w_i
			l_i[i] = new_l_i

			if options.do_weighted_averaging == 1 then
				local alpha = 2/(k + 2)
				wAvg = (1 - alpha) * wAvg + alpha * w 
				lAvg = (1 - alpha) * lAvg + alpha * l 
			end

			k = k + 1	
		end 	-- end of dummy for loop

		if options.debug == 1 then
			local model_debug = {}
			if options.do_weighted_averaging == 1 then
				model_debug.w = wAvg
				model_debug.l = lAvg
			else
				model_debug.w = w
				model_debug.l = l
			end

			local f = helpers.getDual(model_debug.w, model_debug.l, lambda)
			local dualGap2 = helpers.duality_gap(model_debug.w, lambda, model_debug.l, n, options.minitrainset_data, options.minitrainset_label) 
			assert(dualGap2 > 0, 'Duality Gap: ' .. dualGap2)
			local primal = f + dualGap2
--			assert(primal > dualGap2, 'Primal: ' .. primal .. 'Gap: ' .. dualGap2)
			assert(primal > f, 'Primal: ' .. primal .. ' Dual: ' .. (f) .. '\n')
			local training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, model_debug.w)

			io.write(string.format("pass %d (iteration %d), SVM primal = %.8f, SVM dual = %.6f, duality gap = %.6f, training error = %.6f\n", numIter, n*numIter, primal, f, dualGap2, training_loss))
			if dualGap2 < epsilon then
				print("Duality gap below threshold \n")
				print("Current gap = " .. dualGap2 .. "		Gap threshold = " .. epsilon)
				break
			end  -- end of duality gap check if 
			primalArr[numIter] = primal
			dualArr[numIter] = f
			timeArr[numIter] = timer:time().real 
			numIter = numIter + 1
			gnuplot.pdffigure('plots/MinibatchBCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
			gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
			gnuplot.xlabel('Time(s)')
			gnuplot.ylabel('Value')
			gnuplot.plotflush()
		end 	-- end of debug block if 
	end 	-- end of p for loop

	print(string.format("elapsed time: %.2f sec\n", timer:time().real))
	local model_final = {}
	if options.do_weighted_averaging == 1 then
		model_final.w = wAvg
		model_final.l = lAvg
	else
		model_final.w = w
		model_final.l = l
	end

	local training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, model_final.w)
	print("Training loss: " .. training_loss .. "\n")
	t = 0
	iterArr = torch.Tensor(numIter - 1)
	iterArr:apply(function()
		t = t + 1
		return t
	end)
	gnuplot.pdffigure('plots/MinibatchBCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
	gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
	gnuplot.xlabel('Time(s)')
	gnuplot.ylabel('Value')
	gnuplot.plotflush()
	return timeArr, dualArr, numIter
end

return{minibatchBCFW = minibatchBCFW}
