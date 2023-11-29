def pack_to_source_profile_step(pack):
    if pack.startswith("data"):
        return "natural", "natural", "natural"
    if pack.startswith("support"):
        return "support", "support", "support"

    tmp = pack.split(".")
    if tmp[1] == "generate":
        return (tmp[0], tmp[1], tmp[1])
    return (tmp[0], tmp[1], tmp[1]) if len(tmp) == 2 else tmp
