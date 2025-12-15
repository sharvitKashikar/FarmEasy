/* ======================================================
   Startup Performance Analyzer (TypeScript)
   ------------------------------------------------------
   - Tracks revenue & expenses
   - Calculates burn rate
   - Estimates runway
   - Flags financial health
   ====================================================== */

type Currency = "USD" | "INR";

interface FinancialRecord {
  month: string;
  revenue: number;
  expenses: number;
}

interface StartupMetrics {
  burnRate: number;
  runwayMonths: number;
  healthStatus: "SAFE" | "WARNING" | "CRITICAL";
}

class StartupAnalyzer {
  private records: FinancialRecord[] = [];
  private cashInHand: number;
  private currency: Currency;

  constructor(initialCash: number, currency: Currency = "USD") {
    this.cashInHand = initialCash;
    this.currency = currency;
  }

  addMonthlyRecord(record: FinancialRecord): void {
    this.records.push(record);
    console.log(`📅 Added record for ${record.month}`);
  }

  private calculateBurnRate(): number {
    if (this.records.length === 0) return 0;

    const totalBurn = this.records.reduce((sum, r) => {
      return sum + (r.expenses - r.revenue);
    }, 0);

    return totalBurn / this.records.length;
  }

  private calculateRunway(burnRate: number): number {
    if (burnRate <= 0) return Infinity;
    return this.cashInHand / burnRate;
  }

  private evaluateHealth(runway: number): "SAFE" | "WARNING" | "CRITICAL" {
    if (runway >= 12) return "SAFE";
    if (runway >= 6) return "WARNING";
    return "CRITICAL";
  }

  analyze(): StartupMetrics {
    const burnRate = this.calculateBurnRate();
    const runway = this.calculateRunway(burnRate);
    const healthStatus = this.evaluateHealth(runway);

    return {
      burnRate: Math.round(burnRate),
      runwayMonths: runway === Infinity ? Infinity : Math.floor(runway),
      healthStatus,
    };
  }

  generateReport(): void {
    const metrics = this.analyze();

    console.log("\n📊 Startup Financial Report");
    console.log("----------------------------");
    console.log(`💰 Cash in hand: ${this.currency} ${this.cashInHand}`);
    console.log(`🔥 Monthly burn rate: ${this.currency} ${metrics.burnRate}`);
    console.log(
      `⏳ Runway: ${
        metrics.runwayMonths === Infinity
          ? "Profitable (∞)"
          : metrics.runwayMonths + " months"
      }`
    );
    console.log(`🚦 Health Status: ${metrics.healthStatus}`);

    if (metrics.healthStatus === "CRITICAL") {
      console.log("⚠️ ACTION: Reduce costs or raise funds immediately!");
    }
  }
}

/* =====================
   Sample Usage
   ===================== */

const startup = new StartupAnalyzer(250000, "USD");

startup.addMonthlyRecord({
  month: "January",
  revenue: 40000,
  expenses: 70000,
});

startup.addMonthlyRecord({
  month: "February",
  revenue: 50000,
  expenses: 75000,
});

startup.addMonthlyRecord({
  month: "March",
  revenue: 65000,
  expenses: 80000,
});

startup.generateReport();
