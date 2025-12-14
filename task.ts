/* ======================================================
   Task Agent System (TypeScript)
   ------------------------------------------------------
   - Accepts tasks
   - Scores priority
   - Decides actions
   - Executes & logs results
   ====================================================== */

enum TaskType {
  BUG = "BUG",
  FEATURE = "FEATURE",
  REFACTOR = "REFACTOR",
  DOCS = "DOCS",
}

enum TaskStatus {
  PENDING = "PENDING",
  IN_PROGRESS = "IN_PROGRESS",
  DONE = "DONE",
}

interface Task {
  id: string;
  title: string;
  type: TaskType;
  estimatedHours: number;
  createdAt: Date;
  status: TaskStatus;
}

interface DecisionResult {
  taskId: string;
  priorityScore: number;
  decision: "EXECUTE_NOW" | "DEFER" | "IGNORE";
}

class TaskAgent {
  private taskQueue: Task[] = [];

  addTask(task: Task): void {
    this.taskQueue.push(task);
    console.log(`📝 Task added: ${task.title}`);
  }

  private calculatePriority(task: Task): number {
    let score = 0;

    // Task type weight
    switch (task.type) {
      case TaskType.BUG:
        score += 50;
        break;
      case TaskType.FEATURE:
        score += 30;
        break;
      case TaskType.REFACTOR:
        score += 20;
        break;
      case TaskType.DOCS:
        score += 10;
        break;
    }

    // Time-based weight
    if (task.estimatedHours <= 2) score += 20;
    else if (task.estimatedHours <= 5) score += 10;

    // Age-based weight
    const ageInDays =
      (Date.now() - task.createdAt.getTime()) / (1000 * 60 * 60 * 24);

    if (ageInDays > 3) score += 15;

    return score;
  }

  private decide(score: number): "EXECUTE_NOW" | "DEFER" | "IGNORE" {
    if (score >= 70) return "EXECUTE_NOW";
    if (score >= 40) return "DEFER";
    return "IGNORE";
  }

  analyzeTasks(): DecisionResult[] {
    return this.taskQueue.map(task => {
      const priorityScore = this.calculatePriority(task);
      const decision = this.decide(priorityScore);

      return {
        taskId: task.id,
        priorityScore,
        decision,
      };
    });
  }

  execute(): void {
    const decisions = this.analyzeTasks();

    decisions.forEach(decision => {
      const task = this.taskQueue.find(t => t.id === decision.taskId);
      if (!task) return;

      if (decision.decision === "EXECUTE_NOW") {
        task.status = TaskStatus.IN_PROGRESS;
        console.log(`🚀 Executing task: ${task.title}`);

        // simulate work
        setTimeout(() => {
          task.status = TaskStatus.DONE;
          console.log(`✅ Completed: ${task.title}`);
        }, 1000);
      } else if (decision.decision === "DEFER") {
        console.log(`⏳ Deferred: ${task.title}`);
      } else {
        console.log(`❌ Ignored: ${task.title}`);
      }
    });
  }
}

/* =====================
   Sample Usage
   ===================== */

const agent = new TaskAgent();

agent.addTask({
  id: "T1",
  title: "Fix login crash",
  type: TaskType.BUG,
  estimatedHours: 1,
  createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
  status: TaskStatus.PENDING,
});

agent.addTask({
  id: "T2",
  title: "Add dark mode",
  type: TaskType.FEATURE,
  estimatedHours: 6,
  createdAt: new Date(),
  status: TaskStatus.PENDING,
});

agent.addTask({
  id: "T3",
  title: "Update README",
  type: TaskType.DOCS,
  estimatedHours: 1,
  createdAt: new Date(),
  status: TaskStatus.PENDING,
});

agent.execute();
