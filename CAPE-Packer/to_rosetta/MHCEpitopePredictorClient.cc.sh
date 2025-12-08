#!/bin/bash

sed -i 's/Matrix/Client/g' ${1}

sed -i '21a\#include <vector>\
#include <sstream>
37a\#include <stdio.h>\
#include <stdlib.h>\
#include <string.h>\
#include <sys/socket.h>\
#include <arpa/inet.h>\
#include <json-c/json.h> // install: sudo apt-get install libjson-c-dev \
// needs to be called like gcc ./src/others/client.c -o ./src/others/client -ljson-c to include in compilation \
#include <locale.h>
77a\    setlocale(LC_ALL, "en_US.UTF-8");\
    struct sockaddr_in serv_addr;\
\
    // Create a socket\
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {\
        printf("Socket creation error\\n");\
    } else {\
		  // Set up the server address structure\
		  memset(&serv_addr, '\''0'\'', sizeof(serv_addr));\
		  serv_addr.sin_family = AF_INET;\
		  int PORT = std::stoi(fn);\
		  serv_addr.sin_port = htons(PORT);\
\
		  // Convert the host address to binary format\
		  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {\
		      printf("Invalid address/ Address not supported\\n");\
		  } else {\
				// Connect to the server\
				if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {\
				    printf("Connection Failed\\n");\
				}\
		  }\
    }\
    set_peptide_length(9);
78d
83a\void MHCEpitopePredictorClient::set_alleles(std::string alleles)\
{\
    std::cout << "Set alleles: " << alleles << std::endl;\
    std::stringstream ss(alleles);\
    std::string sub_string;\
\
    while (getline(ss, sub_string, '\'','\''))\
    {\
        alleles_.push_back(sub_string);\
    }\
\
    for (auto it=alleles_.begin(); it != alleles_.end(); ++it)\
    {\
        std::cout << *it << std::endl;\
    }\
}
89,100d
109a\	output << "Client predictor; " << alleles_.size() << " alleles; rewards art: " << reward_artificial_ << " nat: " << reward_natural_;
110d
123,127d
128a\    if (pep.size() > 0) {\
		char buffer[1024] = {0};\
		char peptide[256];\
		char allele[256];\
		char rewards[128];\
\
		strcpy(peptide, pep.c_str());\
\
		std::snprintf(rewards, sizeof(rewards), "%f,%f", reward_artificial_, reward_natural_);\
\
		// Convert the request object to a JSON string\
		struct json_object *jobj = NULL;\
		jobj = json_object_new_object();\
		json_object_object_add(jobj, "peptide", json_object_new_string(peptide));\
\
        json_object* alleles_array = json_object_new_array();\
        for (int i = 0; i < alleles_.size(); i++) {\
            strcpy(allele, alleles_[i].c_str());\
            json_object_array_put_idx(alleles_array, i, json_object_new_string(allele));\
        }\
        json_object_object_add(jobj, "alleles", alleles_array);\
\
		json_object_object_add(jobj, "rewards", json_object_new_string(rewards));\
		const char* json_str = json_object_to_json_string(jobj);\
\
		// Send the request to the server\
		send(sock, json_str, strlen(json_str), 0);\
		json_object_put(jobj);\
\
		// Receive the result from the server\
		recv(sock, buffer, 1024, 0);\
		char *eptr;\
		total += strtod(buffer, &eptr);\
	}
131a\void MHCEpitopePredictorClient::set_rewards(core::Real reward_artificial, core::Real reward_natural) {\
	reward_artificial_ = reward_artificial;\
	reward_natural_ = reward_natural;\
}
132,231d
266a\	arc( CEREAL_NVP( reward_artificial_ ) ); // core::Real\
	arc( CEREAL_NVP( reward_natural_ ) ); // core::Real
267,270d
277a\	arc( reward_artificial_ ); // core::Real\
	arc( reward_natural_ ); // core::Real
278,281d
' ${1}

