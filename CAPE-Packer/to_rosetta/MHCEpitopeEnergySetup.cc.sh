#!/bin/bash

sed -i '22a\#include <core/scoring/mhc_epitope_energy/MHCEpitopePredictorClient.hh>
142{N;s/\n/ /}
231a\	MHCEpitopePredictorClientOP pred_client;
285a\	} else if ( method == "Client" ) {\
		firstline >> filename;\
		pred_client = MHCEpitopePredictorClientOP(utility::pointer::make_shared<MHCEpitopePredictorClient>(filename));\
		predictor_ = pred_client;
335a\			if ( method == "Client" ) {\
				std::string alleles;\
				line >> alleles;\
				pred_client->set_alleles(alleles);\
			}
340a\		} else if ( option == "rewards" && method=="Client" ) {\
			// If the threshold keyword is used, set the rewards for the predictor.\
			core::Real reward_artificial;\
			core::Real reward_natural;\
			line >> reward_artificial;\
			line >> reward_natural;\
			pred_client->set_rewards(reward_artificial, reward_natural);
' ${1}

