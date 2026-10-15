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

/**
 * 
 * 
 * 
 * dino q table : 

[["far|high|slow|grounded|standing",[0.15029388710627528,0.051996805675062194,0.0539441560437116]],["far|high|slow|grounded|ducking",[0.04029159553212796,0.027321105929907365,0.01800909537396006]],["far|high|medium|grounded|standing",[0.16489674978278612,0.079099784182441,0.09134648745274025]],["far|high|medium|grounded|ducking",[0.10781001557334512,0.07878329492378335,0.08487522473085544]],["far|high|medium|jumping|standing",[0.09157027397472513,0.09100704164097105,0.11700681432091388]],["far|high|fast|jumping|standing",[0.10784244664335883,0.08329681733492855,0.10165174121528404]],["far|high|fast|grounded|standing",[0.1030966717682804,0.08565159750584558,0.127995452601054]],["far|high|fast|grounded|ducking",[0.10085162358819647,0.10700308532777635,0.10964896735993904]],["far|low|fast|grounded|ducking",[-0.04118186554347042,-0.03347609788324193,-0.01414771192920147]],["far|low|fast|grounded|standing",[-0.06353277004114427,-0.08868051826678164,-0.0588752988860639]],["medium|low|fast|jumping|standing",[-0.6378440474346894,-0.5684973395641209,-0.6393724015965139]],["close|low|fast|jumping|standing",[-1.4565442192570677,-1.4204684313651152,-1.4973699733202899]],["close|low|fast|grounded|standing",[-1.419335646156254,-1.3622316636132252,-1.8842255210169754]],["very-close|low|fast|grounded|ducking",[-5.6953279000000006,-6.12579511,-6.12579511]],["medium|high|fast|grounded|ducking",[0.00788083867752691,0.006434924326292193,0.0028169886826273157]],["medium|high|fast|grounded|standing",[0.013546694536860436,0.0018437645637195445,0.005918530753085161]],["close|high|fast|grounded|ducking",[0.004686850119877032,0.0013946635131864738,0]],["very-close|high|fast|grounded|standing",[-1.533665575011736,-2.4231926661221435,0.009821914597899634]],["very-close|high|fast|jumping|standing",[-1.8895068234938177,-1.3075847945633896,-1.9632971662372016]],["far|high|slow|jumping|standing",[0.03954592232960522,0.021563851634716635,0.06760023119727565]],["medium|low|fast|grounded|ducking",[-0.53021759596556,-0.47552169637563796,-0.520926339530499]],["close|low|fast|grounded|ducking",[-1.4697359765752105,-1.5568365969914613,-1.8313269154661778]],["very-close|low|fast|grounded|standing",[-4.0951,-2.1859257853907237,-5.273091592416231]],["medium|high|fast|jumping|standing",[0.012483077806017547,0.008495421244272876,0.017520185866760626]],["close|high|fast|grounded|standing",[0.011362323359372534,-0.0070283690191660485,0.0033659238784052646]],["far|low|fast|jumping|standing",[-0.10000734558850745,-0.11203754464941551,-0.11290896296128947]],["medium|low|fast|grounded|standing",[-0.5139620410143831,-0.5266143934997384,-0.5017479780553988]],["very-close|low|fast|jumping|standing",[-1.9141127677853982,-3.0222058329861246,-2.6495793541799757]],["close|high|fast|jumping|standing",[-0.17553523013703198,-0.24028005462662355,-0.29193720392549394]],["very-close|high|fast|grounded|ducking",[-0.8061285033351249,-0.032580550067955086,-0.0322072894162404]]]



 dino epsilon : 

 30


 */
