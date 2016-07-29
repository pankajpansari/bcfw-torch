require 'torch'
require 'image'
require 'gnuplot'
matio = require 'matio'
--ProFi = require 'ProFi'
--local mnist = require 'mnist'
local helpers = require("helpers.helperFn")
trainset_size = 10000
debug_flag = 0
data_source = 'pritish'
--torch.manualSeed(123)

local image_size = 32
local nchannel = 3
local nclasses = 10
local featureMapSize
if data_source == 'original' then
	featureMapSize = nclasses * nchannel * image_size * image_size
	featureSize = 3072 
elseif data_source == 'pritish' then
	featureMapSize = nclasses * 4096 
	featureSize = 4096 
end
-- Frank-Wolfe algorithm

--torch.setdefaulttensortype('torch.Tensor')

function FW(minitrainset_data, minitrainset_label, options)
	print("Starting Frank-Wolfe (with line-search)\n")
	local n = trainset_size 
	local w = torch.Tensor(featureMapSize):zero()
	local l = 0
	local K = options.num_passes 
	local lambda = options.lambda
	local ws = torch.Tensor(featureMapSize):zero()
	local ls = 0
	local epsilon = options.epsilon 
	local primalArr = torch.Tensor(K+1):zero()
	local dualArr = torch.Tensor(K+1):zero()
	local numIter = 0
	local gamma = 0
	for k = 0, K do
		ws = ws:zero()
		ls = 0
		for i = 1, n do
			local x = minitrainset_data[i]
			local y = minitrainset_label[i]
			local max_y = helpers.maxOracle(x, y, w)
			local psi_i = helpers.featureVec(x, y) - helpers.featureVec(x, max_y)
			ws = ws + 1/(n*lambda) * psi_i 
			ls = ls + 1/n * helpers.loss(y, max_y)
			assert(helpers.loss(y, max_y) - w:dot(psi_i) >= -1e-12)
		end

		local dualGap = (lambda * w:dot(w - ws) - l + ls)

		io.write(string.format("Duality gap check: gap = %.4f at iteration %d \n", dualGap, k))

		if dualGap < epsilon then
			print("Duality gap below threshold \n")
			break
		end
	
		if options.do_line_search then
			gamma = dualGap/(lambda*(torch.norm(w - ws)^2 + epsilon))
			gamma = math.max(0, math.min(1, gamma))	
	else
			gamma = 2/(k + 2)
		end
		print("Gamma = " .. gamma)
	
		w = (1 - gamma)*w + gamma * ws
		l = (1 - gamma)*l + gamma * ls

		if options.debug == 1 then
--			print("gamma: ", gamma, "\n")
			local f = - helpers.getDual(w, l, lambda)
			local truedualGap = helpers.duality_gap(w, lambda, l, n, minitrainset_data, minitrainset_label) 
			print("Duality gap: " .. truedualGap)
			local primal = - f + truedualGap
			print("Primal (from gap):" .. primal .. "Dual value: " .. f)
			local training_loss = helpers.average_loss(minitrainset_data, minitrainset_label, w)
			print("Training loss: " .. training_loss .. "\n")
			primalArr[k+1] = primal
			dualArr[k+1] = f
			numIter = k + 1
		end

	end
	local training_loss = helpers.average_loss(minitrainset_data, minitrainset_label, w)
	print("Training loss: " .. training_loss .. "\n")
	print(string.format("elapsed time: %.2f sec\n", os.clock() - start_time))
--	t = 0
--	iterArr = torch.Tensor(numIter)
--	iterArr:apply(function()
--		t = t + 1
--		return t
--	end)
--	gnuplot.pdffigure('FW_n' .. n .. '_eps_' .. epsilon .. '.pdf') 
--	gnuplot.plot(
--	{'Primal', iterArr, primalArr[{{1, numIter}}]},
--	{'Dual', iterArr, dualArr[{{1, numIter}}]})
--	gnuplot.xlabel('# iterations')
--	gnuplot.ylabel('Value')
--	gnuplot.plotflush()
end

-- Block-Coordinate Frank-Wolfe algorithm

function BCFW(minitrainset_data, minitrainset_label, options)

	print("Starting Block-Coordinate Frank-Wolfe \n")
--	ProFi:start()
	local n = trainset_size
--	print(featureMapSize)
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
	print("Using " .. n .. " training examples")
	local k = 0
	for p = 1, K do
		for dummy = 1, n do
			local i = dummy
--			local i = n - dummy + 1
			local x = minitrainset_data[i]
			local y = minitrainset_label[i]
			local max_y = helpers.maxOracle(x, y, w)
			local psi_i = helpers.featureVec(x, y) - helpers.featureVec(x, max_y)
			ws = psi_i/(lambda * n)
			ls = helpers.loss(y, max_y)/n
			assert(helpers.loss(y, max_y) - w:dot(psi_i) >= -1e-12)
			local gamma = 0
			if options.do_line_search == 1 then
--				print("Doing line search")
				gamma = (lambda * w:dot(w_i[i] - ws) - l_i[i] + ls)/(lambda * (w_i[i] - ws):dot(w_i[i] - ws) + epsilon)
				if gamma < 0 then
					gamma = 0
				elseif gamma > 1 then
					gamma = 1
				end
			else
--				print("Not doing line search")
				gamma = 2*n/(k + 2*n)	
			end
--			gamma = tonumber(string.format("%.2f", gamma))
			
--			print("Gamma: " .. gamma)
--			print("max oracle val: " .. max_y)
--			local dummy_training_loss = helpers.average_loss(minitrainset_data, minitrainset_label, w)
--			print("Training loss: " .. dummy_training_loss .. "\n")
--			if dummy == 20 then
--				break
--			end

			local new_w_i = (1 - gamma)*w_i[i] + gamma * ws
			local new_l_i = (1 - gamma)*l_i[i] + gamma * ls

			w = w + new_w_i - w_i[i]
			l = l + new_l_i - l_i[i]
		
			w_i[i] = new_w_i
			l_i[i] = new_l_i
			k = k + 1	
		end -- end dummy for loop

		if options.debug == 1 then
			local f = helpers.getDual(w, l, lambda)
			local dualGap2 = helpers.duality_gap(w, lambda, l, n, minitrainset_data, minitrainset_label) 
			local primal = f + dualGap2
			assert(primal > f, 'Primal: ' .. primal .. 'Dual: ' .. (f) .. '\n')
			local training_loss = helpers.average_loss(minitrainset_data, minitrainset_label, w)

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
			gnuplot.pdffigure('BCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
			gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
			gnuplot.xlabel('Time(s)')
			gnuplot.ylabel('Value')
			gnuplot.plotflush()
--		       	
	
--		elseif p % 10 == 0 then
--			local f = helpers.getDual(w, l, lambda)
--			local dualGap2 = helpers.duality_gap(w, lambda, l, n, minitrainset_data, minitrainset_label) 
--			io.write(string.format("Duality gap check: gap = %.4f 	at pass %d (iteration %d) \n", dualGap2, k, k*trainset_size))
--			io.write(string.format("Time: %.2f (s) 	 Dual value: %.6f \n" , timer:time().real, f))
--			if dualGap2 < epsilon then
--				print("Duality gap below threshold \n")
--				print("Current gap = " .. dualGap2 .. "		Gap threshold = " .. epsilon)
--				break
--			end
--			dualArr[numIter] = f
--			timeArr[numIter] = timer:time().real 
--			numIter = numIter + 1
		end

--	break
end -- end k for loop


	print(string.format("elapsed time: %.2f sec\n", timer:time().real))
	local training_loss = helpers.average_loss(minitrainset_data, minitrainset_label, w)
	print("Training loss: " .. training_loss .. "\n")
	t = 0
	iterArr = torch.Tensor(numIter - 1)
	iterArr:apply(function()
		t = t + 1
		return t
	end)
	gnuplot.pdffigure('BCFW_CIFAR_n' .. n .. '_eps_' .. epsilon .. '_lambda_' .. lambda .. '.pdf') 
--	gnuplot.plot(
--	{'Primal', iterArr, primalArr[{{1, numIter}}]},
--	{'Dual', iterArr, dualArr[{{1, numIter}}]})
	gnuplot.plot({'Dual', timeArr[{{1, numIter - 1}}], dualArr[{{1, numIter - 1}}], '-'} )
	gnuplot.xlabel('Time(s)')
	gnuplot.ylabel('Value')
	gnuplot.plotflush()
--	ProFi:stop()
--	ProFi:writeReport('BCFW_profile_sparse2.txt')
end

function main()
--	local trainset = mnist.traindataset()
--	local testset = mnist.testdataset()

	cmd = torch.CmdLine()
	cmd:option('-lambda', 0.1, 'learning rate parameter')
	params = cmd:parse(arg)

	local minitrainset_data
	local minitrainset_label
	print("Loading data")
	if data_source == 'original' then
		local trainset = torch.load('cifar.torch/cifar10-train.t7')
		minitrainset_data = torch.Tensor(trainset_size, nchannel, image_size, image_size)
		minitrainset_label = torch.Tensor(trainset_size)
		for i = 1, trainset_size do
			local id = i
			minitrainset_data[i] = trainset.data[id]
			minitrainset_label[i] = trainset.label[id]
		end
	elseif data_source == 'pritish' then
--			local trainset = matio.load('CIFAR-10_FeatureVec/trial.mat')
			local trainset = matio.load('CIFAR-10_FeatureVec/trainData.mat')
			minitrainset_data = torch.Tensor(trainset_size, 4096)
			minitrainset_label = torch.Tensor(trainset_size)
			for i = 1, trainset_size do
				local id = i
				minitrainset_data[i] = trainset.trainFeatures[id]
				minitrainset_label[i] = trainset.trainLabels[id]
			end
	end

	--print("max label : " .. torch.max(minitrainset_label) .. "  min label: " .. torch.min(minitrainset_label))
	-- change to zero mean and 1 st dev
--	meanVal = torch.mean(minitrainset_data)
--	stdVal = torch.std(minitrainset_data)

--	minitrainset_data = (minitrainset_data - meanVal)/stdVal

	--sanity check
--	print("After normalization:\n")
--	print("mean: " .. torch.mean(minitrainset_data) .. "std deviation: " .. torch.std(minitrainset_data) .. "\n")

	-- check the distribution of labels
	--initialize
--	local labelVec = {}
--	for i = 1, 10 do
--		labelVec[i] = 0
--	end
--
--	-- add entry 
--	for i = 1, trainset_size do
--		labelVec[minitrainset_label[i] + 1] = labelVec[minitrainset_label[i] + 1] + 1
--	end
--
--	-- print distribution
--
--	print("Label distribution of training set\n")
--
--	for i = 1, 10 do
--		print(i - 1, ": ", labelVec[i]*100/trainset_size)
--	end

	local options = {} 
	options.num_passes = 10000

	options.do_line_search = 1
	options.do_weighted_averaging = 0
	options.time_budget = inf
	options.debug = 1
	options.lambda = params.lambda
	options.epsilon = 0.01		--gap threshold
	print(options)

--	FW(minitrainset_data, minitrainset_label, options)
	print("*************************************************************")
	timer = torch.Timer()	
	BCFW(minitrainset_data, minitrainset_label, options)
end

main()
