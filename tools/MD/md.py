#!/usr/bin/env python

from openmm.app import *
from openmm import *
from openmm.unit import *
from dcdsubreporter import DCDSubReporter

# from https://github.com/gpantel/MD_methods-and-analysis/tree/master/OpenMM_DCDSubset
# https://gpantel.github.io/computational-method/OpenMM_DCDSubset/

import mdtraj as md
from mdtraj.reporters import NetCDFReporter

import hashlib
from sys import stdout
import argparse
import pdb
from datetime import datetime
import numpy as np
import math
import os
import sys
import pandas as pd
import yaml


def now(format="%Y-%m-%d %H:%M:%S"):
    """returns the current time in the given format

    :param format: format of the time string (default: '%Y-%m-%d %H:%M:%S')
    :return: current time
    """

    return datetime.now().strftime(format)


def get_md_param_hash(params=None, yaml_file_path=None):
    if yaml_file_path is not None:
        with open(yaml_file_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
    hash_text = ""
    hash_text += f"Solvent: {params['solvent']}\n"
    hash_text += f"Temperature: {params['temperature']:.5f} Kelvin\n"
    hash_text += f"Step size: {params['step_size']} femtoseconds\n"
    hash_object = hashlib.sha256()
    hash_object.update(hash_text.encode("utf-8"))
    hash_code = hash_object.hexdigest()
    return hash_code[:6]


def rename_simulations(directory_path):
    # List all subdirectories recursively
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(directory_path):
        subdirectories.extend([os.path.join(dirpath, d) for d in dirnames])

    # Print the list of subdirectories
    for subdir in subdirectories:
        yaml_file_path = os.path.join(subdir, "args.yaml")
        if os.path.exists(yaml_file_path):
            directory = os.path.join(*subdir.split("/")[:-1])
            parameter_hash = get_md_param_hash(yaml_file_path=yaml_file_path)
            new_dir = os.path.join(os.sep, directory, parameter_hash)
            os.rename(subdir, new_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--pdb",
        type=str,
        help="the pdb file to simulate. if non is provided the output subfolders will be renamed using the current parameter hash",
    )
    argparser.add_argument("--output", type=str, default=".", help="the output folder")
    argparser.add_argument(
        "--solvent", type=str, default="implicit", help="implicit/explicit"
    )
    argparser.add_argument(
        "--temperature", type=float, default=25.0, help="temperature in degrees Celsius"
    )
    argparser.add_argument(
        "--time_span",
        type=float,
        default=2000,
        help="simulation time span in nanoseconds",
    )
    argparser.add_argument(
        "--step_size", type=float, default=2, help="step-size in femtoseconds"
    )
    argparser.add_argument(
        "--freq_report",
        type=float,
        default=2000,
        help="report state every freq_report steps",
    )
    argparser.add_argument(
        "--num_pdbs",
        type=int,
        default=10,
        help="how many structures to write to the output.pdb",
    )
    argparser.add_argument("--rmsd", action="store_true")
    argparser.add_argument(
        "--resume",
        action="store_true",
        help="continue the simulation from a checkpoint",
    )
    argparser.add_argument(
        "--pbar",
        type=int,
        default=100,
        help="how many progress bar updates should there be",
    )
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--full", action="store_true")
    args = argparser.parse_args()

    if args.debug:
        pdb.set_trace()

    if args.pdb is None:
        print(f"Rename subfolders of {args.pdb}")
        rename_simulations(args.output)
        sys.exit()

    temperature = (
        args.temperature + 273.15
    )  # converts the degrees Celsius into degrees Kelvin
    freq_report = args.freq_report
    steps = math.ceil(
        (args.time_span * nanoseconds).value_in_unit(femtoseconds) / args.step_size
    )

    structure_name = args.pdb.split(os.path.sep)[-1].removesuffix(".pdb")
    structure_folder = os.path.join(args.output, structure_name)
    os.makedirs(structure_folder, exist_ok=True)

    parameter_hash = get_md_param_hash(params=vars(args))

    output_folder = os.path.join(structure_folder, f"{parameter_hash}")
    if os.path.exists(output_folder) and not args.resume:
        sys.exit()

    os.makedirs(output_folder, exist_ok=True)
    print(f"write output to folder: {output_folder}")

    with open(os.path.join(output_folder, "args.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    print(f"Start: {now()}")

    print("Set up the forcefield")
    if args.solvent == "explicit":
        forcefield = ForceField("amber99sb.xml", "tip3p.xml")
        friction = 1 / picosecond
    elif args.solvent == "implicit":
        forcefield = ForceField("amber99sb.xml", "amber99_obc.xml")
        friction = 91 / picosecond

    print("load the pdb file")
    pdb_structure = PDBFile(args.pdb)

    print("add missing hydrogens and the solvent")
    modeller = Modeller(pdb_structure.topology, pdb_structure.positions)
    modeller.addHydrogens(forcefield)
    if args.solvent == "explicit":
        modeller.addSolvent(forcefield, model="tip3p", padding=1.0 * nanometers)

    print("create the system")
    model = modeller
    system = forcefield.createSystem(model.topology, constraints=HBonds)
    with open(os.path.join(output_folder, "system.xml"), "w") as file:
        file.write(XmlSerializer.serialize(system))

    print("prepare the simulation")
    integrator = LangevinIntegrator(
        temperature * kelvin, friction, args.step_size * femtoseconds
    )
    simulation = Simulation(model.topology, system, integrator)
    if args.resume:
        print("load checkpoint")
        simulation.loadCheckpoint(os.path.join(output_folder, "checkpoint.ckpt"))
        continue_simulation = True
    else:
        print("set positions")
        simulation.context.setPositions(model.positions)
        continue_simulation = False
    simulation.minimizeEnergy()

    # Find the alpha carbon atoms
    alpha_carbon_atom_indices = [
        atom.index for atom in simulation.topology.atoms() if atom.name == "CA"
    ]  # atom.residue.name in aa3]  # remove HOH (water), NA (sodium)

    print("add reporters")
    simulation.reporters.append(
        CheckpointReporter(
            os.path.join(output_folder, "checkpoint.ckpt"), int(steps / args.num_pdbs)
        )
    )

    if args.full:
        simulation.reporters.append(
            DCDReporter(
                os.path.join(output_folder, "output_full.dcd"),
                freq_report,
                append=continue_simulation,
            )
        )
    else:
        simulation.reporters.append(
            DCDSubReporter(
                os.path.join(output_folder, "output_sub.dcd"),
                freq_report,
                atom_indices=alpha_carbon_atom_indices,
                append=continue_simulation,
            )
        )
    # simulation.reporters.append(NetCDFReporter(os.path.join(output_folder, 'output.netcdf'), freq_report, atomSubset=alpha_carbon_atom_indices, append=continue_simulation))

    simulation.reporters.append(
        PDBReporter(
            os.path.join(output_folder, "output.pdb"), int(steps / args.num_pdbs)
        )
    )

    reporter = StateDataReporter(
        os.path.join(output_folder, "report.tsv"),
        int(steps / 1000),
        append=continue_simulation,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        separator="\t",
    )
    simulation.reporters.append(reporter)

    print(
        f"simulate {steps:,} steps using {simulation.context.getPlatform().getName()} platform"
    )
    trajectory_snapshots = []

    start = datetime.now()
    for step in range(steps):
        simulation.step(1)

        if step % (steps // args.pbar) == 0:
            diff = (datetime.now() - start).total_seconds()
            pc = step / steps
            eta = (diff / pc - diff) / 60 if pc > 0 else -1
            print(f"\rProgress: {pc:.2%}", end="")
            if eta > 60:
                print(f", ETA: {int(eta // 60)}:{int(eta % 60)}", end="")
            else:
                print(f", ETA: {eta:.0f} min", end="")
            print(f"{'':10}", end="")

        if step % freq_report == 0 or step == (steps - 1):
            if args.rmsd:
                state = simulation.context.getState(getPositions=True)
                positions = state.getPositions()
                trajectory_snapshots.append(positions.value_in_unit(nanometer))

    print("")

    # calculating RMSD
    if args.rmsd:
        trajectory_positions = np.array(trajectory_snapshots)
        trajectory = md.Trajectory(trajectory_positions, simulation.topology)

        rmsd = md.rmsd(
            target=trajectory,
            reference=trajectory,
            frame=0,
            atom_indices=alpha_carbon_atom_indices,
        )
        with open(os.path.join(output_folder, "rmsd"), "w") as file:
            file.write("\n".join([f"{r:.3f}" for r in rmsd]))

    # print(rmsd)
    print(f"Stop: {now()}")
