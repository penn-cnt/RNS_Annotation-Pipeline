from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KMeansSamplingGPU, KCenterGreedy, KCenterGreedyPCA, BALDDropout, \
    AdversarialBIM, AdversarialDeepFool, VarRatio, MeanSTD, BadgeSampling, CEALSampling, \
    LossPredictionLoss, VAAL, WAAL, \
    LeastConfidenceRNS, LeastConfidenceDropoutRNS, EntropySamplingRNS, EntropySamplingDropoutRNS, MarginSamplingRNS, \
    MarginSamplingDropoutRNS, BALDDropoutRNS, MeanSTDRNS, KMeansSamplingRNS, KCenterGreedyRNS, KCenterGreedyPCARNS, \
    BadgeSamplingRNS, WAALRNS, LossPredictionLossRNS ,RandomSamplingRNS


def get_strategy(STRATEGY_NAME, dataset, net, args_input, args_task, rns_data=False):
    if rns_data == False:
        if STRATEGY_NAME == 'RandomSampling':
            return RandomSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'LeastConfidence':
            return LeastConfidence(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'MarginSampling':
            return MarginSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'EntropySampling':
            return EntropySampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'LeastConfidenceDropout':
            return LeastConfidenceDropout(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'MarginSamplingDropout':
            return MarginSamplingDropout(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'EntropySamplingDropout':
            return EntropySamplingDropout(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'KMeansSampling':
            return KMeansSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'KMeansSamplingGPU':
            return KMeansSamplingGPU(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'KCenterGreedy':
            return KCenterGreedy(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'KCenterGreedyPCA':
            return KCenterGreedyPCA(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'BALDDropout':
            return BALDDropout(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'VarRatio':
            return VarRatio(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'MeanSTD':
            return MeanSTD(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'BadgeSampling':
            return BadgeSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'LossPredictionLoss':
            return LossPredictionLoss(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'AdversarialBIM':
            return AdversarialBIM(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'AdversarialDeepFool':
            return AdversarialDeepFool(dataset, net, args_input, args_task)
        elif 'CEALSampling' in STRATEGY_NAME:
            return CEALSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'VAAL':
            net_vae, net_disc = get_net_vae(args_task['name'])
            handler_joint = get_handler_joint(args_task['name'])
            return VAAL(dataset, net, args_input, args_task, net_vae=net_vae, net_dis=net_disc,
                        handler_joint=handler_joint)
        elif STRATEGY_NAME == 'WAAL':
            return WAAL(dataset, net, args_input, args_task)
        else:
            raise NotImplementedError
    else:
        if STRATEGY_NAME == 'RandomSampling':
            return RandomSamplingRNS(dataset, net, args_input, args_task) #Done
        elif STRATEGY_NAME == 'LeastConfidence':
            return LeastConfidenceRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'MarginSampling':
            return MarginSamplingRNS(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'EntropySampling':
            return EntropySamplingRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'LeastConfidenceDropout':
            return LeastConfidenceDropoutRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'MarginSamplingDropout':
            return MarginSamplingDropoutRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'EntropySamplingDropout':
            return EntropySamplingDropoutRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'KMeansSampling':
            return KMeansSamplingRNS(dataset, net, args_input, args_task)#Done
        # elif STRATEGY_NAME == 'KMeansSamplingGPU':
        #     return KMeansSamplingGPU(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'KCenterGreedy':
            return KCenterGreedyRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'KCenterGreedyPCA':
            return KCenterGreedyPCARNS(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'BALDDropout':
            return BALDDropoutRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'VarRatio':
            return VarRatio(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'MeanSTD':
            return MeanSTDRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'BadgeSampling':
            return BadgeSamplingRNS(dataset, net, args_input, args_task)#Done
        elif STRATEGY_NAME == 'LossPredictionLoss':
            return LossPredictionLossRNS(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'AdversarialBIM':
            return AdversarialBIM(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'AdversarialDeepFool':
            return AdversarialDeepFool(dataset, net, args_input, args_task)
        elif 'CEALSampling' in STRATEGY_NAME:
            return CEALSampling(dataset, net, args_input, args_task)
        elif STRATEGY_NAME == 'VAAL':
            net_vae, net_disc = get_net_vae(args_task['name'])
            handler_joint = get_handler_joint(args_task['name'])
            return VAAL(dataset, net, args_input, args_task, net_vae=net_vae, net_dis=net_disc,
                        handler_joint=handler_joint)
        elif STRATEGY_NAME == 'WAAL':
            return WAALRNS(dataset, net, args_input, args_task)
        else:
            raise NotImplementedError
