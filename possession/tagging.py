"""What the possession logic takes from the smart tagging file and the ATD metadata.

- kick-off tags: (kind, start frame, end frame), kind = "Kick Off (Start)" / "Kick Off (After Goal)".
  Tag times are seconds from the start of the video; frame = round(seconds * FPS).
- sides: which end HOME defends after each "Kick Off (Start)" tag (a half starts there), from the mean x of
  each team's players at the tag's first frame (at a kick-off every player stands in his own half).
- goalkeeper slots: the ATD slots whose position_type is Goalkeeper (one per team).
"""
import xml.etree.ElementTree as ET

import numpy as np

import possession as P

START, AFTER_GOAL = "Kick Off (Start)", "Kick Off (After Goal)"


def kickoffs(path):
    """[(kind, a, b)] in time order, a / b = first / last frame of the tag."""
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        if inst.findtext("code") == "SET PIECES" and labels.get("Type") in (START, AFTER_GOAL):
            out.append((labels["Type"], round(float(inst.findtext("start")) * P.FPS),
                        round(float(inst.findtext("end")) * P.FPS)))
    return sorted(out, key=lambda k: k[1])


def goalkeeper_slots(meta_path):
    """Slot indices (same order as baseline.team_of_each_track) of the ATD goalkeepers."""
    root = ET.parse(meta_path).getroot()
    team_ids = {team.get("id") for team in root.iter("Team")}
    slots = [p for p in root.iter("Player") if p.get("teamId") in team_ids]
    return [s for s, p in enumerate(slots)
            if any(q.findtext("Name") == "position_type" and q.findtext("Value") == "Goalkeeper"
                   for q in p.iter("ProviderParameter"))]


def sides(kicks, atd_path, teams):
    """[(first frame of the half, HOME defends left?)] for each "Kick Off (Start)" tag."""
    out = []
    for kind, a, _ in kicks:
        if kind != START:
            continue
        players, _ = P.read_atd(atd_path, a, a)
        x = players[0, :, 0]
        home, away = (np.nanmean(np.where(teams == tm, x, np.nan)) for tm in (P.HOME, P.AWAY))
        out.append((a, bool(home < away)))
    return out


def home_left(frames, half_sides):
    """Per frame: HOME defends the left end (bool), from the latest half start at or before it."""
    out = np.zeros(len(frames), bool)
    for a, left in half_sides:
        out[frames >= a] = left
    return out
