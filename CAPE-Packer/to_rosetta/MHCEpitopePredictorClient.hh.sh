#!/bin/bash

sed -i 's/Matrix/Client/g' ${1}

sed -i '26a\#include <vector>
89a\	void set_rewards(core::Real reward_artificial, core::Real reward_natural);\
	void set_alleles(std::string alleles);\
	int sock = 0;
90,93d
95a\	core::Real reward_artificial_ = 1.;\
	core::Real reward_natural_ = 1.;\
	std::vector<std::string> alleles_;
96,103d' ${1}

