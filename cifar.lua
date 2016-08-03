require 'torch'
require 'image'
require 'gnuplot'
matio = require 'matio'
--ProFi = require 'ProFi'
--local mnist = require 'mnist'
local helpers = require("helpers.helperFn")
local solverFW = require("solvers.solverFW")
local solverBCFW = require("solvers.solverBCFW")
local solverminiBCFW = require("solvers.solverminiBCFW")
eps = torch.pow(2, -52)
trainset_size = 50
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

--torch.setdefaulttensortype('torch.Tensor')

cmd = torch.CmdLine()
cmd:option('-lambda', 0.1, 'learning rate parameter')
cmd:option('-method', 4, 'algorithm')
params = cmd:parse(arg)

local minitrainset_data
local minitrainset_label
print("Loading data")
if data_source == 'original' then
	local trainset = torch.load('data/cifar.torch/cifar10-train.t7')
	minitrainset_data = torch.Tensor(trainset_size, nchannel, image_size, image_size)
	minitrainset_label = torch.Tensor(trainset_size)
	for i = 1, trainset_size do
		local id = i
		minitrainset_data[i] = trainset.data[id]
		minitrainset_label[i] = trainset.label[id]
	end
elseif data_source == 'pritish' then
--			local trainset = matio.load('CIFAR-10_FeatureVec/trial.mat')
		local trainset = matio.load('data/CIFAR-10_FeatureVec/trainData.mat')
		print(trainset.trainFeatures:size())
		print(trainset.trainLabels:size())
		minitrainset_data = torch.Tensor(trainset_size, 4096)
		minitrainset_label = torch.Tensor(trainset_size)
		for i = 1, trainset_size do
			local id = i
			minitrainset_data[i] = trainset.trainFeatures[id]
			minitrainset_label[i] = trainset.trainLabels[id]
		end
end

local options = {} 
options.num_passes = 10000

options.trainset_size = trainset_size
options.debug_flag = debug_flag
options.nclasses = nclasses
options.featureSize = featureSize
options.minitrainset_data = minitrainset_data
options.minitrainset_label = minitrainset_label
options.do_line_search = 1
options.do_weighted_averaging = 1
options.time_budget = inf
options.debug = 1
options.lambda = params.lambda
options.epsilon = 0.1		--gap threshold

print(options)
timer = torch.Timer()	

local dualArr1 = torch.Tensor(options.num_passes+1):zero()
local timeArr1 = torch.Tensor(options.num_passes+1):zero()
local numIter1 = 0 
local dualArr2 = torch.Tensor(options.num_passes+1):zero()
local timeArr2 = torch.Tensor(options.num_passes+1):zero()
local numIter2 = 0 
local dualArr3 = torch.Tensor(options.num_passes+1):zero()
local timeArr3 = torch.Tensor(options.num_passes+1):zero()
local numIter3 = 0 

if params.method == 0 then
	timeArr1, dualArr1, numIter1 = solverFW.FW(options)
elseif params.method == 1 then
	timeArr2, dualArr2, numIter2 = solverBCFW.BCFW(options)
elseif params.method == 2 then
	timeArr3, dualArr3, numIter3 = solverminiBCFW.minibatchBCFW(options, 10)
elseif params.method == 3 then
--	timeArr1, dualArr1, numIter1 = solverFW.FW(options)
	timer = torch.Timer()
	timeArr2, dualArr2, numIter2 = solverBCFW.BCFW(options)
	timer = torch.Timer()
	timeArr3, dualArr3, numIter3 = solverminiBCFW.minibatchBCFW(options, 10)
else
	print("Use -method option: 0 for FW, 1 for BCFW, 2 for minibatch BCFW")
end

print("Plotting")
gnuplot.pdffigure('plots/comparison.pdf') 
gnuplot.plot(
--{'FW', timeArr1[{{1, numIter1 - 1}}], dualArr1[{{1, numIter1 - 1}}], '-'},
{'BCFW', timeArr2[{{1, numIter2 - 1}}], dualArr2[{{1, numIter2 - 1}}], '-'},
{'miniBCFW', timeArr3[{{1, numIter3 - 1}}], dualArr3[{{1, numIter3 - 1}}], '-'}
)
gnuplot.xlabel('Time(s)')
gnuplot.ylabel('Value')
gnuplot.plotflush()
