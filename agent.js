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
  };

  runner.playIntro();

  // Main agent loop: observe -> decide -> act.

  setInterval(() => {
    try {
      const runner = Runner.instance_;

      // 1. Ensure the game engine exists.
      if (!runner) {
        return;
      }

      // 2. Handle crashes before checking the paused state.
      if (runner.crashed) {
        runner.restart();

        runner.paused = false;

        return;
      }

      // 3. Don't make decisions while the game is inactive.
      if (!runner.playing || runner.paused) {
        return;
      }

      // 4. Observe the current game state.
      const state = getState();

      // 5. Decide what the AI should do.
      const action = chooseAction(state);

      // 6. Log jump decisions for debugging.
      if (action === ACTIONS.JUMP) {
       
      }

      // 7. Execute the chosen action.
      applyAction(action);
    } catch (error) {
      console.error("AI controller error:", error);
    }
  }, 1000 / 60);
});
