// -*- mode:c++;tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
// vi: set ts=2 noet:
//
// (c) Copyright Rosetta Commons Member Institutions.
// (c) This file is part of the Rosetta software suite and is made available under license.
// (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
// (c) For more information, see http://www.rosettacommons.org. Questions about this can be
// (c) addressed to University of Washington CoMotion, email: license@uw.edu.

/// @file core/scoring/mhc_epitope_energy/MHCEpitopePredictorClient.cc
/// @brief MHC epitope predictor using a position weight matrix, targeted to Propred though in theory generalizable to others
/// @author Chris Bailey-Kellogg, cbk@cs.dartmouth.edu; Brahm Yachnin, brahm.yachnin@rutgers.edu

#include <core/types.hh>
#include <basic/Tracer.hh>
#include <utility/exit.hh>
#include <utility/io/izstream.hh>
#include <utility/file/FileName.hh>
#include <basic/database/open.hh>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <utility/string_util.hh>
#include <core/chemical/AA.hh>

#include <core/scoring/mhc_epitope_energy/MHCEpitopePredictorClient.hh>
#include <core/scoring/ScoringManager.hh>

#ifdef    SERIALIZATION
// Utility serialization headers
#include <utility/vector1.srlz.hh>
#include <utility/serialization/serialization.hh>

// Cereal headers
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/map.hpp>
#endif // SERIALIZATION


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <json-c/json.h> // install: sudo apt-get install libjson-c-dev
// needs to be called like gcc ./src/others/client.c -o ./src/others/client -ljson-c to include in compilation
#include <locale.h>

//#define PORT 12345



namespace core {
namespace scoring {
namespace mhc_epitope_energy {

static basic::Tracer TR("core.scoring.mhc_epitope_energy.MHCEpitopePredictorClient");

AlleleClient::AlleleClient()
{}

AlleleClient::AlleleClient(std::string name, utility::vector1< Real > threshes, PWM profile)
: name_(name), threshes_(threshes), profile_(profile)
{}

AlleleClient::~AlleleClient()
{}

bool AlleleClient::operator==(AlleleClient const &other)
{
	// TODO: does name matter?
	return profile_ == other.profile_ && threshes_ == other.threshes_;
}

bool AlleleClient::is_hit(std::string const &pep, Real thresh)
{
	Real total(0);
	// Loop over the profile of this allele (i.e. each position).
	for ( core::Size p=1; p<=profile_.size(); p++ ) {
		// Look up the appropriate weight in the profile for the peptide's AA at this position, and add it to total.
		total += profile_[p][pep[p-1]];
	}

	// If the total for this peptide is greater than the threshold, it is a hit. Return whether ot not it is.
	return total >= threshes_[(core::Size)thresh];
}

MHCEpitopePredictorClient::MHCEpitopePredictorClient()
{}

MHCEpitopePredictorClient::MHCEpitopePredictorClient( std::string const &fn )
{
    setlocale(LC_ALL, "en_US.UTF-8");

    struct sockaddr_in serv_addr;

    // Create a socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket creation error\n");
    } else {
		  // Set up the server address structure
		  memset(&serv_addr, '0', sizeof(serv_addr));
		  serv_addr.sin_family = AF_INET;
		  int PORT = std::stoi(fn);
		  serv_addr.sin_port = htons(PORT);

		  // Convert the host address to binary format
		  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
		      printf("Invalid address/ Address not supported\n");
		  } else {
				// Connect to the server
				if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
				    printf("Connection Failed\n");
				}
		  }
    }
    set_peptide_length(9);
}

MHCEpitopePredictorClient::~MHCEpitopePredictorClient()
{}

void MHCEpitopePredictorClient::set_alleles(std::string alleles)
{
    std::cout << "Set alleles: " << alleles << std::endl;
    std::stringstream ss(alleles);
    std::string sub_string;

    while (getline(ss, sub_string, ','))
    {
        alleles_.push_back(sub_string);
    }

    for (auto it=alleles_.begin(); it != alleles_.end(); ++it)
    {
        std::cout << *it << std::endl;
    }
}

bool MHCEpitopePredictorClient::operator==(MHCEpitopePredictor const &other)
{
	MHCEpitopePredictorClient const *o = dynamic_cast<MHCEpitopePredictorClient const *>(&other);
	if ( !o ) return false;

	return true;
}

std::string MHCEpitopePredictorClient::report() const
{
	std::stringstream output("");

	// TODO: more (allele names, ...)?
	output << "Client predictor; " << alleles_.size() << " alleles; rewards art: " << reward_artificial_ << " nat: " << reward_natural_;

	return output.str();
}

core::Real MHCEpitopePredictorClient::score(std::string const &pep)
{
    // std::cout << "MHCEpitopePredictorClient::score start ^" << pep  << "$" << std::endl;
	if ( pep.size() != get_peptide_length() ) {
		TR.Error << "Scoring peptide of size " << pep.size() << " with a matrix expecting peptides of size " << get_peptide_length() << std::endl;
		utility_exit_with_message("MHCEpitopePredictorClient is trying to score a peptide of the incorrect size!");
	}

	core::Real total(0);

    if (pep.size() > 0) {
		char buffer[1024] = {0};
		char peptide[256];
		char allele[256];
		char rewards[128];

		strcpy(peptide, pep.c_str());

		std::snprintf(rewards, sizeof(rewards), "%f,%f", reward_artificial_, reward_natural_);

        // std::cout << "Send peptide: " << peptide << std::endl;
		// Convert the request object to a JSON string
		struct json_object *jobj = NULL;
		jobj = json_object_new_object();
		json_object_object_add(jobj, "peptide", json_object_new_string(peptide));

        json_object* alleles_array = json_object_new_array();
        for (int i = 0; i < alleles_.size(); i++) {
            strcpy(allele, alleles_[i].c_str());
            json_object_array_put_idx(alleles_array, i, json_object_new_string(allele));
        }
        json_object_object_add(jobj, "alleles", alleles_array);

		json_object_object_add(jobj, "rewards", json_object_new_string(rewards));
		const char* json_str = json_object_to_json_string(jobj);

        // std::cout << "Send request to server" << std::endl;
		// Send the request to the server
		send(sock, json_str, strlen(json_str), 0);
		json_object_put(jobj);

        // std::cout << "Wait for result" << std::endl;
		// Receive the result from the server
		recv(sock, buffer, 1024, 0);
		// std::cout << "Received result" << std::endl;
		char *eptr;
		total += strtod(buffer, &eptr);

	}
	return total;
}

/// @brief Sets the threshold for what is considered to be an epitope -- top thresh% of peptides in this implementation
/// @details Includes error checking in the propred matrix case to make sure we get a reasonable threshold
void MHCEpitopePredictorClient::set_rewards(core::Real reward_artificial, core::Real reward_natural) {
	reward_artificial_ = reward_artificial;
	reward_natural_ = reward_natural;
}

}//ns mhc_epitope_energy
}//ns scoring
}//ns core

#ifdef    SERIALIZATION

/// @brief Automatically generated serialization method
template< class Archive >
void
core::scoring::mhc_epitope_energy::AlleleClient::save( Archive & arc ) const {
	arc( CEREAL_NVP( name_ ) ); // std::string
	arc( CEREAL_NVP( threshes_ ) ); // core::vector1
	arc( CEREAL_NVP( profile_ ) ); // PWM
}

/// @brief Automatically generated deserialization method
template< class Archive >
void
core::scoring::mhc_epitope_energy::AlleleClient::load( Archive & arc ) {
	arc( name_ ); // std::string
	arc( threshes_ ); // core::vector1
	arc( profile_ ); // PWM
}

SAVE_AND_LOAD_SERIALIZABLE( core::scoring::mhc_epitope_energy::AlleleClient );
CEREAL_REGISTER_TYPE( core::scoring::mhc_epitope_energy::AlleleClient )

CEREAL_REGISTER_DYNAMIC_INIT( core_scoring_mhc_epitope_energy_AlleleClient )

/// @brief Automatically generated serialization method
template< class Archive >
void
core::scoring::mhc_epitope_energy::MHCEpitopePredictorClient::save( Archive & arc ) const {
	arc( cereal::base_class< core::scoring::mhc_epitope_energy::MHCEpitopePredictor >( this ) );
	arc( CEREAL_NVP( reward_artificial_ ) ); // core::Real
	arc( CEREAL_NVP( reward_natural_ ) ); // core::Real
}

/// @brief Automatically generated deserialization method
template< class Archive >
void
core::scoring::mhc_epitope_energy::MHCEpitopePredictorClient::load( Archive & arc ) {
	arc( cereal::base_class< core::scoring::mhc_epitope_energy::MHCEpitopePredictor >( this ) );
	arc( reward_artificial_ ); // core::Real
	arc( reward_natural_ ); // core::Real
}

SAVE_AND_LOAD_SERIALIZABLE( core::scoring::mhc_epitope_energy::MHCEpitopePredictorClient );
CEREAL_REGISTER_TYPE( core::scoring::mhc_epitope_energy::MHCEpitopePredictorClient )

CEREAL_REGISTER_DYNAMIC_INIT( core_scoring_mhc_epitope_energy_MHCEpitopePredictorClient )
#endif // SERIALIZATION
