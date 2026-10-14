window.addEventListener("load", () => {
  // Start the game automatically
  Runner.instance_.playIntro();

  // Run the AI loop at 60 FPS
  setInterval(() => {
    const runner = Runner.instance_;
    if (!runner || runner.crashed || !runner.horizon.obstacles.length) {
      if (runner && runner.crashed) {
        runner.restart(); // Auto-restart on crash for training
      }
      return;
    }

    const obs = runner.horizon.obstacles[0];
    const distance = obs.xPos - runner.tRex.xPos;
    const speed = runner.currentSpeed;

    // Simple physics calculation before adding neural networks
    const timeToCollision = distance / speed;

    if (timeToCollision > 0 && timeToCollision < 18) {
      // Jump if obstacle is at ground level
      if (obs.yPos > 50) {
        /**
         * For example, if the game canvas is 150 pixels tall:
            - yPos = 10: near the top.
            - yPos = 50: closer to the middle.
            - yPos = 100: closer to the bottom.

            so ypos > 50 means the collision obj is nearer to ground, so action -> duck
         */
        document.dispatchEvent(
          new KeyboardEvent("keydown", { keyCode: 32, which: 32 }),
        );
      }
    } else {
      document.dispatchEvent(
        new KeyboardEvent("keyup", { keyCode: 32, which: 32 }),
      );
    }
  }, 1000 / 60);
});
