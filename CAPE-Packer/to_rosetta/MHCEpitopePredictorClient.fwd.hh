// -*- mode:c++;tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
// vi: set ts=2 noet:
//
// (c) Copyright Rosetta Commons Member Institutions.
// (c) This file is part of the Rosetta software suite and is made available under license.
// (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
// (c) For more information, see http://www.rosettacommons.org. Questions about this can be
// (c) addressed to University of Washington CoMotion, email: license@uw.edu.

/// @file core/scoring/mhc_epitope_energy/MHCEpitopePredictorClient.fwd.hh
/// @brief forward declaration for class MHCEpitopePredictorClient
/// @details Follows analogous file for Vikram K. Mulligan's NetChargeEnergy
/// @author Chris Bailey-Kellogg, cbk@cs.dartmouth.edu; Brahm Yachnin, brahm.yachnin@rutgers.edu

#ifndef INCLUDED_core_scoring_mhc_epitope_energy_MHCEpitopePredictorClient_fwd_hh
#define INCLUDED_core_scoring_mhc_epitope_energy_MHCEpitopePredictorClient_fwd_hh

#include <utility/pointer/owning_ptr.hh>

namespace core {
namespace scoring {
namespace mhc_epitope_energy {

class MHCEpitopePredictorClient;
typedef utility::pointer::shared_ptr< MHCEpitopePredictorClient > MHCEpitopePredictorClientOP;
typedef utility::pointer::shared_ptr< MHCEpitopePredictorClient const > MHCEpitopePredictorClientCOP;

} // mhc_epitope_energy
} // scoring
} // core

#endif
