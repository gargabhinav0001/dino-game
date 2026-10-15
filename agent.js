window.addEventListener("load", () => {
  const runner = Runner.instance_;

  if (!runner) {
    console.error("Dino AI: Runner instance not found.");
    return;
  }

  // Action IDs: these will become the agent's action space.
  const ACTIONS = {
    NONE: 0,
    JUMP: 1,
    DUCK: 2,
  };

  let episodeNumber = 0;
  let bestDistance = 0;
  let totalDistance = 0;

  const agent = new QLearningAgent([ACTIONS.NONE, ACTIONS.JUMP, ACTIONS.DUCK], {
    alpha: 0.1,
    gamma: 0.99,
    epsilon: 1.0,
    epsilonMin: 0.05,
    epsilonDecay: 0.995,
  });

  // Stores the previous decision for learning.
  let previousState = null;
  let previousAction = null;

  let lastDecisionTime = 0;
  const DECISION_INTERVAL_MS = 100;

  // Convert your game state into the format QLearningAgent expects.
  function toAgentState(state) {
    return {
      ...state,
      jumping: Boolean(state.isJumping),
      ducking: Boolean(state.isDucking),
    };
  }

  // 1. OBSERVATION: collect the current game state.
  function getState() {
    const obstacles = runner.horizon.obstacles;
    const trex = runner.tRex;
    const obs = obstacles[0];

    const speed = runner.currentSpeed;
    const maxSpeed = runner.config.MAX_SPEED || 13;
    const gameWidth = runner.dimensions.WIDTH || 600;

    // Normalized values are easier to use in many ML models.
    const distance = obs
      ? obs.xPos - (trex.xPos + trex.config.WIDTH)
      : gameWidth;

    const timeToCollision = obs ? (obs.xPos - trex.xPos) / speed : Infinity;

    const nextDistance = obstacles[1]
      ? obstacles[1].xPos - trex.xPos
      : gameWidth;

    return {
      distance,
      timeToCollision,
      speed,
      obstacleY: obs ? obs.yPos : 0,
      obstacleWidth: obs ? obs.width : 0,
      isJumping: trex.jumping ? 1 : 0,
      isDucking: trex.ducking ? 1 : 0,

      // Fixed-scale numerical features for future learning.
      features: [
        distance / gameWidth,
        (obs ? obs.yPos : 0) / 150,
        (obs ? obs.width : 0) / 100,
        speed / maxSpeed,
        trex.jumping ? 1 : 0,
        trex.ducking ? 1 : 0,
        nextDistance / gameWidth,
      ],
    };
  }

  // 2. POLICY: choose an action.
  // For now, preserve your existing rule-based behavior.

  function chooseAction(state) {
    if (!Number.isFinite(state.timeToCollision)) {
      return ACTIONS.NONE;
    }

    // Start jumping earlier to give the Dino time to rise.
    if (
      state.timeToCollision > 0 &&
      state.timeToCollision <= 45 &&
      state.obstacleY > 50
    ) {
      return ACTIONS.JUMP;
    }

    return ACTIONS.NONE;
  }

  function dispatchKey(type, key, code, keyCode) {
    document.dispatchEvent(
      new KeyboardEvent(type, {
        key,
        code,
        keyCode,
        which: keyCode,
        bubbles: true,
      }),
    );
  }

  // 3. ACTION EXECUTION: send the selected action to the game.

  let lastAction = 0;

  function applyAction(action) {
    const trex = runner.tRex;

    // Don't interfere with the game while it is crashed.
    if (runner.crashed) return;

    if (action === ACTIONS.JUMP) {
      // Start a jump only if the dinosaur is on the ground.
      if (!trex.jumping && !trex.ducking) {
        trex.startJump(runner.currentSpeed);
      }
    } else if (action === ACTIONS.DUCK) {
      if (!trex.jumping && !trex.ducking) {
        trex.setDuck(true);
      }
    } else {
      // Release ducking, but let an existing jump finish naturally.
      if (trex.ducking) {
        trex.setDuck(false);
      }
    }
  }

  // Expose the interface for debugging in DevTools.
  window.dinoAgent = {
    ACTIONS,
    getState,
    chooseAction,

    // Learning diagnostics
    agent,
    getLearnedStateCount: () => agent.qTable.size,
    getEpsilon: () => agent.epsilon,
    getQTable: () => Array.from(agent.qTable.entries()),
  };

  runner.playIntro();

  // Main agent loop: observe -> decide -> act.

  setInterval(() => {
    try {
      const runner = Runner.instance_;

      if (!runner) return;

      // A crash ends the current learning episode.

      if (runner.crashed) {
        // 1. Penalize the last action for ending in a crash.
        if (previousState !== null && previousAction !== null) {
          agent.learn(previousState, previousAction, -10, previousState, true);
        }

        // 2. Record performance BEFORE restarting the game.
        const distance = runner.distanceRan;

        episodeNumber++;
        totalDistance += distance;
        bestDistance = Math.max(bestDistance, distance);

        // 3. Finish this learning episode.
        agent.finishEpisode();

        console.log("AI episode performance", {
          episode: episodeNumber,
          distance: Math.round(distance),
          averageDistance: Math.round(totalDistance / episodeNumber),
          bestDistance: Math.round(bestDistance),
          epsilon: Number(agent.epsilon.toFixed(4)),
          learnedStates: agent.qTable.size,
        });

        // 4. Clear the previous decision.
        previousState = null;
        previousAction = null;

        // 5. Restart the game.
        runner.restart();
        runner.paused = false;

        return;
      }

      if (!runner.playing || runner.paused) return;

      // Limit AI decisions to once every 100 milliseconds.
      const now = performance.now();

      if (now - lastDecisionTime < DECISION_INTERVAL_MS) {
        return;
      }

      lastDecisionTime = now;

      // 1. Observe the current game.
      const rawState = getState();

      const state = toAgentState(rawState);

      // 2. Learn from the previous action's transition.
      if (previousState !== null && previousAction !== null) {
        agent.learn(
          previousState,
          previousAction,
          0.01, // Small survival reward
          state,
          false,
        );
      }

      // 3. Choose an action using Q-learning.
      const action = agent.chooseAction(state);

      // 4. Execute that action.
      applyAction(action);

      // 5. Remember this decision for the next learning update.
      previousState = state;
      previousAction = action;
    } catch (error) {
      console.error("AI controller error:", error);
    }
  }, 1000 / 60);
});
