-- Block-Coordinate Frank-Wolfe algorithm
local eps = torch.pow(2, -52)
helpers = require 'helpers.helperFn'

local function BCFW(options)

	print("Starting Block-Coordinate Frank-Wolfe \n")
	local featureSize = options.featureSize
	local featureMapSize = options.nclasses * featureSize
	local n = options.trainset_size
	local debug_flag = options.debug_flag
	local w = torch.Tensor(featureMapSize):zero()
	local w_i = torch.Tensor(n, featureMapSize):zero()
	local ws = torch.Tensor(featureMapSize):zero()
	local l = 0
	local l_i = torch.Tensor(n):zero()
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
	for p = 1, K do
		for dummy = 1, n do
			local i = dummy
--			local i = n - dummy + 1
			local x = options.minitrainset_data[i]
			local y = options.minitrainset_label[i]
			local max_y = helpers.maxOracle(x, y, w)
			local psi_i = helpers.featureVec(x, y) - helpers.featureVec(x, max_y)
			ws = psi_i/(lambda * n)
			ls = helpers.loss(y, max_y)/n
			assert(helpers.loss(y, max_y) - w:dot(psi_i) >= -1e-12)
			local gamma = 0
			if options.do_line_search == 1 then
			--	print("Doing line search")
				gamma = (w:dot(w_i[i] - ws) - 1/lambda * (l_i[i] - ls))/((w_i[i] - ws):dot(w_i[i] - ws) + eps)
				if gamma < 0 then
					gamma = 0
				elseif gamma > 1 then
					gamma = 1
				end
--				print("Gamma: " .. gamma .. "formula: " ..  2*n/(k + 2*n))
				
			else
			--	print("Not doing line search")
				gamma = 2*n/(k + 2*n)	
	--			gamma = tonumber(string.format("%.2f", gamma))
			end
			
--			print("max oracle val: " .. max_y)
--			local dummy_training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, w)
--			print("Training loss: " .. dummy_training_loss .. "\n")
--			if dummy == 20 then
--				break
--			end

--			print("Gamma: " .. gamma)
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
		end -- end dummy for loop
		
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
			local primal = f + dualGap2
			assert(primal > f, 'Primal: ' .. primal .. 'Dual: ' .. (f) .. '\n')
			local training_loss = helpers.average_loss(options.minitrainset_data, options.minitrainset_label, model_debug.w)

			io.write(string.format("pass %d (iteration %d), SVM primal = %.6f, SVM dual = %.6f, duality gap = %.6f, training error = %.6f\n", numIter, n*numIter, primal, f, dualGap2, training_loss))
			if dualGap2 < epsilon then
				print("Duality gap below threshold \n")
				print("Current gap = " .. dualGap2 .. "		Gap threshold = " .. epsilon)
				break
			end
			primalArr[numIter] = primal
			dualArr[numIter] = f
			timeArr[numIter] = timer:time().real 
			numIter = numIter + 1
			gnuplot.pdffigure('plots/BCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
			gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
			gnuplot.xlabel('Time(s)')
			gnuplot.ylabel('Value')
			gnuplot.plotflush()
--		       	
	
		end
	end -- end k for loop


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
	gnuplot.pdffigure('plots/BCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
--	gnuplot.plot(
--	{'Primal', iterArr, primalArr[{{1, numIter}}]},
--	{'Dual', iterArr, dualArr[{{1, numIter}}]})
	gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
	gnuplot.xlabel('Time(s)')
	gnuplot.ylabel('Value')
	gnuplot.plotflush()
--	ProFi:stop()
--	ProFi:writeReport('BCFW_profile_sparse2.txt')
	local returnTable = {}
	returnTable.timeArr = timeArr
	returnTable.dualArr = dualArr
	returnTable.numIter = numIter
	return timeArr, dualArr, numIter 
end

return{BCFW = BCFW}
