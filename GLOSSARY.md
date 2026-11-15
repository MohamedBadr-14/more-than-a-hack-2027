# Possession labelling

Labelling every video frame of the BAR-ATH match with the team in possession, or dead ball, from tracking, video and audio.

## Labels and spells

**Possession**:
The team the ball belongs to while it is in play; it stays with the last team that controlled it through loose and contested phases.
_Avoid_: ownership, control (as a label)

**Spell**:
A maximal run of frames labelled with one team; the unit the ranking metric (spellF1) matches.
_Avoid_: possession run, stint

**Dead ball (D)**:
The stretch from the moment play stops to the restart kick, when the ball is not in play.
_Avoid_: out of play, idle

## Ball events

**Touch**:
A moment the ball's path changes (turn, slow-down, kick from slow), attributed to the team of the nearest player.
_Avoid_: contact, event (alone)

**Real kick**:
A touch that kicks a slow ball away fast ("kick from slow"); the only thing that ends a pause after a whistle or the ball going out.

**Hold**:
One player staying within reach of the ball for at least a second with no opponent within reach.
_Avoid_: dribble, carry

**Controlled sequence**:
Consecutive touches/holds of one team, unbroken by the other team, with at least two touches or a hold.
_Avoid_: control, possession phase

**Bounce**:
A touch whose velocity change lies along the camera's line of sight; flagged for review, still counted.

**Ball out**:
The ball seen beyond a touchline or goal line by our own detector.

**Feet**:
A ball position inside a crowded penalty box, taken to be the tracker on players' boots rather than the ball.

## Stoppages and restarts

**Play stopped**:
The exact moment the ball becomes dead (ball over a line, end of the referee's whistle, ball coming to rest); where dead ball begins.
_Avoid_: whistle (alone), break

**Pause**:
Players walking for at least two seconds together with a stoppage sign (ball still, crowded end, whistled, outed); the reliable but late evidence that play stopped.
_Avoid_: wait (old name in code), idle period

**Crowded end**:
Many players between one goal line and the 16.5 m line across the full width; the corner shape.

**Whistled**:
Players slowing down shortly after a referee whistle.

**Outed**:
Players slowing down after the ball was seen out, with no real kick in between.

**Referee whistle**:
Two or more flat tones starting together at the referee's whistle pitches; fans' whistles glide and are ignored.

**Restart**:
The first real kick after a pause; where dead ball ends.
_Avoid_: set piece (that is the tagging's word)

**Restart team**:
The team of the restart kicker; it has possession from the restart, whoever receives the kick.

**Kick-off**:
The restart from the centre spot at the start of each half and after a goal.
