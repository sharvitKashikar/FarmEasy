# Startup Performance Analyzer

class Startup:
    def __init__(self, name, funding_million, monthly_users, monthly_revenue):
        self.name = name
        self.funding_million = funding_million
        self.monthly_users = monthly_users
        self.monthly_revenue = monthly_revenue

    def revenue_per_user(self):
        return round(self.monthly_revenue / self.monthly_users, 2)

    def health_score(self):
        score = 0

        if self.funding_million >= 5:
            score += 30
        if self.monthly_users >= 10000:
            score += 40
        if self.monthly_revenue >= 50000:
            score += 30

        return score

    def summary(self):
        print(f"\n📊 Startup: {self.name}")
        print(f"Funding: ${self.funding_million}M")
        print(f"Monthly Users: {self.monthly_users}")
        print(f"Monthly Revenue: ${self.monthly_revenue}")
        print(f"Revenue/User: ${self.revenue_per_user()}")
        print(f"Health Score: {self.health_score()}/100")


# Sample usage
startup1 = Startup(
    name="QuickPay",
    funding_million=8,
    monthly_users=15000,
    monthly_revenue=72000
)

startup2 = Startup(
    name="EduSpark",
    funding_million=2,
    monthly_users=6000,
    monthly_revenue=25000
)

startup1.summary()
startup2.summary()
