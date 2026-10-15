
class QLearningAgent {
  constructor(actions, options = {}) {
    this.actions = actions;
    this.qTable = new Map();

    this.alpha = options.alpha ?? 0.1;
    this.gamma = options.gamma ?? 0.99;

    // Start by exploring; reduce exploration as learning progresses.
    this.epsilon = options.epsilon ?? 1.0;
    this.epsilonMin = options.epsilonMin ?? 0.05;
    this.epsilonDecay = options.epsilonDecay ?? 0.995;
  }

  // Convert continuous game data into manageable state categories.
  getStateKey(state) {
    const ttc = state.timeToCollision;

    const distanceBucket =
      !Number.isFinite(ttc) ? "far" :
      ttc < 10 ? "very-close" :
      ttc < 20 ? "close" :
      ttc < 40 ? "medium" : "far";

    const heightBucket =
      state.obstacleY > 80 ? "low" : "high";

    const speedBucket =
      state.speed < 8 ? "slow" :
      state.speed < 13 ? "medium" : "fast";

    return [
      distanceBucket,
      heightBucket,
      speedBucket,
      state.jumping ? "jumping" : "grounded",
      state.ducking ? "ducking" : "standing"
    ].join("|");
  }

  getQValues(stateKey) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(
        stateKey,
        this.actions.map(() => 0)
      );
    }

    return this.qTable.get(stateKey);
  }

  chooseAction(state) {
    const key = this.getStateKey(state);

    // Exploration: sometimes try a random action.
    if (Math.random() < this.epsilon) {
      return this.actions[
        Math.floor(Math.random() * this.actions.length)
      ];
    }

    // Exploitation: choose the action with the highest Q-value.
    const values = this.getQValues(key);
    const bestValue = Math.max(...values);

    const bestIndices = values
      .map((value, index) => ({ value, index }))
      .filter(item => item.value === bestValue)
      .map(item => item.index);

    const index =
      bestIndices[Math.floor(Math.random() * bestIndices.length)];

    return this.actions[index];
  }

  learn(state, action, reward, nextState, done = false) {
    const stateKey = this.getStateKey(state);
    const nextKey = this.getStateKey(nextState);

    const currentValues = this.getQValues(stateKey);
    const actionIndex = this.actions.indexOf(action);

    if (actionIndex === -1) {
      throw new Error("Unknown action: " + action);
    }

    const currentQ = currentValues[actionIndex];

    const nextBestQ = done
      ? 0
      : Math.max(...this.getQValues(nextKey));

    // Q-learning update rule.
    const target = reward + this.gamma * nextBestQ;

    currentValues[actionIndex] =
      currentQ + this.alpha * (target - currentQ);
  }

  finishEpisode() {
    this.epsilon = Math.max(
      this.epsilonMin,
      this.epsilon * this.epsilonDecay
    );
  }
}

window.QLearningAgent = QLearningAgent;
