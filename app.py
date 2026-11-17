from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import re
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-development-only-change-before-deploy")

# SQLite database location. Locally this creates instance/career_predictor.db.
DATABASE_PATH = os.environ.get(
    "DATABASE_PATH", str(Path(app.root_path) / "instance" / "career_predictor.db")
)


def get_db_connection():
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                skills TEXT,
                cgpa REAL,
                interests TEXT,
                projects TEXT,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                user_id INTEGER PRIMARY KEY,
                skills TEXT NOT NULL DEFAULT '',
                cgpa REAL NOT NULL DEFAULT 0,
                interests TEXT DEFAULT '',
                projects TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)



init_db()

CAREER_DOMAINS = {
    "Artificial Intelligence & ML": {
        "tagline": "Build intelligent systems, predictive models, and neural networks.",
        "keywords": "python machine learning deep learning neural networks tensorflow pytorch nlp computer vision mathematics statistics algorithms ai",
        "required_skills": ["Python", "Linear Algebra & Stats", "Scikit-Learn", "TensorFlow / PyTorch", "NLP / Computer Vision", "Model Deployment"],
        "min_cgpa": 7.5,
        "roadmap": [
            {
                "step": "1",
                "phase": "Month 1 · Core Foundations",
                "focus": "Python, NumPy, Pandas & Math",
                "tasks": "Master data manipulation with Pandas/NumPy and refresh linear algebra, probability, and statistics.",
                "milestone": "Mini-Project: Exploratory Data Analysis Notebook"
            },
            {
                "step": "2",
                "phase": "Month 2 · Classical Machine Learning",
                "focus": "Scikit-Learn & Predictive Modeling",
                "tasks": "Learn regression, classification, clustering, and model evaluation metrics using Scikit-Learn.",
                "milestone": "Mini-Project: House Price or Disease Predictor"
            },
            {
                "step": "3",
                "phase": "Month 3 · Deep Learning & Neural Nets",
                "focus": "PyTorch / TensorFlow, CNNs & RNNs",
                "tasks": "Build neural networks for image recognition and sequence modeling.",
                "milestone": "Mini-Project: Image Classifier or Sentiment Analyzer"
            },
            {
                "step": "4",
                "phase": "Month 4 · Specialization & Deployment",
                "focus": "LLMs / Computer Vision + FastAPI",
                "tasks": "Pick a specialization track, wrap your model in an API, and deploy a live web demo.",
                "milestone": "Capstone: End-to-End Deployed AI App"
            }
        ]
    },
    "Full-Stack Web Development": {
        "tagline": "Design and ship complete, scalable web applications from UI to database.",
        "keywords": "html css javascript react nodejs express mongodb sql frontend backend api web development git typescript nextjs",
        "required_skills": ["HTML5 & CSS3", "JavaScript (ES6+)", "React / Next.js", "Node.js & Express", "SQL / MongoDB", "Git & REST APIs"],
        "min_cgpa": 6.5,
        "roadmap": [
            {
                "step": "1",
                "phase": "Month 1 · Modern Frontend Essentials",
                "focus": "HTML5, Responsive CSS & JavaScript ES6+",
                "tasks": "Master Flexbox, CSS Grid, DOM manipulation, async/await, and Fetch API.",
                "milestone": "Mini-Project: Interactive Task Manager"
            },
            {
                "step": "2",
                "phase": "Month 2 · Component UI Frameworks",
                "focus": "React.js, State Management & Tailwind",
                "tasks": "Build reusable components, manage state with Hooks, and implement client-side routing.",
                "milestone": "Mini-Project: E-Commerce Storefront UI"
            },
            {
                "step": "3",
                "phase": "Month 3 · Backend APIs & Databases",
                "focus": "Node.js, Express & PostgreSQL / MongoDB",
                "tasks": "Design RESTful APIs, handle CRUD operations, and structure relational/NoSQL schemas.",
                "milestone": "Mini-Project: REST API with JWT Auth"
            },
            {
                "step": "4",
                "phase": "Month 4 · Full-Stack Capstone & Cloud",
                "focus": "Authentication, Testing & Deployment",
                "tasks": "Connect frontend + backend, add role-based auth, and deploy on Vercel/Render.",
                "milestone": "Capstone: Full-Stack SaaS Web Application"
            }
        ]
    },
    "Cloud & DevOps Engineering": {
        "tagline": "Automate software delivery and architect reliable cloud infrastructure.",
        "keywords": "aws azure gcp docker kubernetes linux networking ci cd jenkins terraform bash cloud computing virtualization",
        "required_skills": ["Linux Administration", "Networking Basics", "AWS / Azure Core", "Docker", "Kubernetes", "CI/CD (GitHub Actions / Jenkins)"],
        "min_cgpa": 7.0,
        "roadmap": [
            {
                "step": "1",
                "phase": "Month 1 · Linux & Networking Core",
                "focus": "Linux CLI, Bash Scripting & TCP/IP",
                "tasks": "Master server administration, permissions, SSH, DNS, HTTP/HTTPS, and shell automation.",
                "milestone": "Mini-Project: Automated Server Health Bash Script"
            },
            {
                "step": "2",
                "phase": "Month 2 · Cloud Infrastructure",
                "focus": "AWS / Azure Core Services",
                "tasks": "Configure compute (EC2), storage (S3), networking (VPC), IAM roles, and managed databases.",
                "milestone": "Mini-Project: Host a Scalable 2-Tier Web App on Cloud"
            },
            {
                "step": "3",
                "phase": "Month 3 · Containers & Orchestration",
                "focus": "Docker & Kubernetes",
                "tasks": "Write multi-stage Dockerfiles, compose microservices, and orchestrate pods on Kubernetes.",
                "milestone": "Mini-Project: Containerized Microservices Stack"
            },
            {
                "step": "4",
                "phase": "Month 4 · CI/CD & Infrastructure as Code",
                "focus": "GitHub Actions, Jenkins & Terraform",
                "tasks": "Build zero-downtime deployment pipelines and provision cloud resources using Terraform.",
                "milestone": "Capstone: Automated CI/CD + Terraform Pipeline"
            }
        ]
    },
    "Cybersecurity": {
        "tagline": "Protect networks, applications, and data against modern cyber threats.",
        "keywords": "network security ethical hacking cryptography linux kali penetration testing wireshark firewalls vulnerability owasp security",
        "required_skills": ["Computer Networks", "Linux & Bash", "Web Security (OWASP Top 10)", "Cryptography", "Penetration Testing", "SIEM / Wireshark"],
        "min_cgpa": 7.0,
        "roadmap": [
            {
                "step": "1",
                "phase": "Month 1 · Networking & OS Internals",
                "focus": "TCP/IP, Wireshark & Linux Security",
                "tasks": "Analyze network packets, understand ports/protocols, and harden Linux systems.",
                "milestone": "Mini-Project: Network Traffic & Packet Audit Report"
            },
            {
                "step": "2",
                "phase": "Month 2 · Application & Crypto Security",
                "focus": "OWASP Top 10, Firewalls & Cryptography",
                "tasks": "Learn SQL injection, XSS, CSRF defenses, TLS/SSL, hashing, and encryption standards.",
                "milestone": "Mini-Project: Vulnerable Web App Audit"
            },
            {
                "step": "3",
                "phase": "Month 3 · Ethical Hacking Labs",
                "focus": "Kali Linux, Nmap & Burp Suite",
                "tasks": "Practice reconnaissance, vulnerability scanning, and exploitation on TryHackMe/HackTheBox.",
                "milestone": "Milestone: Complete 15+ CTF / Lab Rooms"
            },
            {
                "step": "4",
                "phase": "Month 4 · Blue Team & Certification Prep",
                "focus": "SIEM Monitoring (Splunk) & Security+",
                "tasks": "Learn log analysis, incident response workflows, and prepare for CompTIA Security+.",
                "milestone": "Capstone: Security Operations Lab & Incident Playbook"
            }
        ]
    },
    "Data Science & Analytics": {
        "tagline": "Turn raw data into actionable business insights and visual stories.",
        "keywords": "data analysis sql python pandas numpy power bi tableau statistics excel visualization big data analytics",
        "required_skills": ["Python (Pandas/NumPy)", "SQL", "Statistics & Probability", "Data Visualization (PowerBI/Tableau)", "Exploratory Data Analysis", "Excel"],
        "min_cgpa": 7.0,
        "roadmap": [
            {
                "step": "1",
                "phase": "Month 1 · SQL & Python Data Wrangling",
                "focus": "Advanced SQL & Pandas",
                "tasks": "Master SQL joins, CTEs, window functions, and clean messy datasets with Python Pandas.",
                "milestone": "Mini-Project: E-Commerce Sales SQL Analysis"
            },
            {
                "step": "2",
                "phase": "Month 2 · Applied Statistics & EDA",
                "focus": "Hypothesis Testing & A/B Analysis",
                "tasks": "Learn distributions, correlation, p-values, and uncover hidden trends in real datasets.",
                "milestone": "Mini-Project: Customer Churn Statistical Study"
            },
            {
                "step": "3",
                "phase": "Month 3 · Business Intelligence & Dashboards",
                "focus": "Power BI / Tableau & Storytelling",
                "tasks": "Design executive KPI dashboards with interactive filters and clear visual hierarchy.",
                "milestone": "Mini-Project: Interactive Revenue BI Dashboard"
            },
            {
                "step": "4",
                "phase": "Month 4 · Predictive Business Modeling",
                "focus": "Forecasting & Kaggle Case Studies",
                "tasks": "Combine regression/classification models with BI dashboards for end-to-end decision support.",
                "milestone": "Capstone: Full Business Analytics Portfolio Case"
            }
        ]
    }
}

JOB_ROLES = [
    {"title":"Python Developer","category":"Software Development","skills":["python","flask","django","api","sql","git"],"description":"Build backend services, APIs and automation tools."},
    {"title":"Junior Data Analyst","category":"Data & Analytics","skills":["sql","excel","python","pandas","power bi","tableau","statistics"],"description":"Turn business data into reports, dashboards and insights."},
    {"title":"Frontend Developer","category":"Web Development","skills":["html","css","javascript","react","typescript","ui"],"description":"Create responsive interfaces and accessible web experiences."},
    {"title":"Full-Stack Developer","category":"Web Development","skills":["html","css","javascript","react","node.js","express","sql","api"],"description":"Develop both browser interfaces and server-side application features."},
    {"title":"Junior AI/ML Engineer","category":"AI & Machine Learning","skills":["python","machine learning","scikit-learn","pandas","numpy","statistics","deep learning"],"description":"Prepare data, build models and evaluate machine-learning solutions."},
    {"title":"Cloud / DevOps Associate","category":"Cloud & DevOps","skills":["linux","aws","azure","docker","kubernetes","git","ci/cd","networking"],"description":"Support cloud infrastructure, deployments, automation and monitoring."},
    {"title":"Cybersecurity Analyst (Trainee)","category":"Cybersecurity","skills":["networking","linux","cybersecurity","wireshark","owasp","python","security"],"description":"Help monitor systems, review vulnerabilities and improve security controls."},
    {"title":"QA / Test Automation Engineer","category":"Software Quality","skills":["python","java","selenium","testing","api","sql","javascript"],"description":"Test applications, report defects and automate repeatable checks."},
    {"title":"Database / SQL Developer (Trainee)","category":"Data & Backend","skills":["sql","mysql","postgresql","sqlite","database","python"],"description":"Design queries, maintain data structures and support application data."},
    {"title":"Business Intelligence Analyst","category":"Analytics","skills":["power bi","tableau","excel","sql","data visualization","statistics"],"description":"Build dashboards and communicate measurable business performance."}
]

def find_similar_jobs(skills_text, interests_text, projects_text):
    text = f"{skills_text} {interests_text} {projects_text}".lower()
    found = set(re.findall(r"[a-zA-Z0-9+#./-]+", text))
    matches = []
    for job in JOB_ROLES:
        matched = [skill for skill in job["skills"] if skill.lower() in text or skill.lower() in found]
        # Give directly entered skills more weight; avoid presenting zero-fit roles.
        score = round((len(matched) / len(job["skills"])) * 100)
        if score > 0:
            matches.append({"title":job["title"],"category":job["category"],"description":job["description"],"match_score":score,"matched_skills":matched,"skills_to_learn":[s for s in job["skills"] if s not in matched][:4]})
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches[:6]

def analyze_profile(skills_text, cgpa, interests_text, projects_text):
    user_corpus = f"{skills_text} {skills_text} {interests_text} {projects_text} {projects_text}".lower()
    
    domain_names = list(CAREER_DOMAINS.keys())
    domain_docs = [CAREER_DOMAINS[d]["keywords"] for d in domain_names]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([user_corpus] + domain_docs)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    scores = {}
    for idx, domain in enumerate(domain_names):
        base_score = similarities[idx] * 100
        cgpa_bonus = 5.0 if cgpa >= CAREER_DOMAINS[domain]["min_cgpa"] else 0.0
        scores[domain] = round(min(base_score * 2.2 + cgpa_bonus, 98.0), 1)
        if scores[domain] < 15.0:
            scores[domain] = round(15.0 + (idx * 2.5), 1)

    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_career = sorted_domains[0][0]
    best_score = sorted_domains[0][1]
    
    user_skills_lower = skills_text.lower()
    required = CAREER_DOMAINS[best_career]["required_skills"]
    
    matched_skills = []
    missing_skills = []
    for skill in required:
        tokens = [t.strip() for t in skill.lower().replace("(", "").replace(")", "").split("/")]
        if any(token in user_skills_lower for token in tokens):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)
    
    min_req_cgpa = CAREER_DOMAINS[best_career]["min_cgpa"]
    if cgpa >= min_req_cgpa:
        cgpa_feedback = f"CGPA {cgpa} meets the typical {min_req_cgpa}+ shortlist benchmark."
    else:
        cgpa_feedback = f"CGPA {cgpa} is below {min_req_cgpa} — offset this with 2+ deployed projects."

    readiness_pct = round((len(matched_skills) / len(required)) * 100)

    return {
        "best_career": best_career,
        "tagline": CAREER_DOMAINS[best_career]["tagline"],
        "match_score": best_score,
        "readiness_pct": readiness_pct,
        "all_scores": dict(sorted_domains),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "cgpa_feedback": cgpa_feedback,
        "roadmap": CAREER_DOMAINS[best_career]["roadmap"],
        "similar_jobs": find_similar_jobs(skills_text, interests_text, projects_text)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/status")
def status():
    user_email = session.get("user")
    if user_email:
        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT email, name FROM users WHERE email = ?", (user_email,)
            ).fetchone()
        if user:
            return jsonify({"logged_in": True, "email": user["email"], "name": user["name"]})
    return jsonify({"logged_in": False})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    name = data.get("name", "").strip() or email.split("@")[0].title()
    is_register = data.get("is_register", False)

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"success": False, "message": "Enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "message": "Use a password with at least 8 characters."}), 400
    if is_register and not data.get("name", "").strip():
        return jsonify({"success": False, "message": "Enter your name to create an account."}), 400

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash, name FROM users WHERE email = ?", (email,)
        ).fetchone()

        if is_register:
            if user:
                return jsonify({"success": False, "message": "Account already exists. Please sign in."}), 400
            conn.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (email, generate_password_hash(password), name)
            )
            user = conn.execute(
                "SELECT id, email, name FROM users WHERE email = ?", (email,)
            ).fetchone()
        else:
            if not user or not check_password_hash(user["password_hash"], password):
                return jsonify({"success": False, "message": "Invalid email or password."}), 401

    session["user"] = user["email"]
    return jsonify({"success": True, "email": user["email"], "name": user["name"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True})

@app.route("/predict", methods=["POST"])
def predict():
    user_email = session.get("user")
    if not user_email:
        return jsonify({"error": "AUTH_REQUIRED", "message": "Please log in first."}), 401

    data = request.get_json(silent=True) or {}
    skills = str(data.get("skills", "")).strip()
    interests = str(data.get("interests", "")).strip()
    projects = str(data.get("projects", "")).strip()
    try:
        cgpa = float(data.get("cgpa", 0))
    except (TypeError, ValueError):
        return jsonify({"error":"INVALID_CGPA", "message":"Enter a valid CGPA between 0 and 10."}), 400
    if not skills:
        return jsonify({"error":"SKILLS_REQUIRED", "message":"Please enter at least one technical skill."}), 400
    if not 0 <= cgpa <= 10:
        return jsonify({"error":"INVALID_CGPA", "message":"CGPA must be between 0 and 10."}), 400

    result = analyze_profile(skills, cgpa, interests, projects)

    # Save each prediction under the logged-in student's database ID.
    with get_db_connection() as conn:
        user = conn.execute("SELECT id FROM users WHERE email = ?", (user_email,)).fetchone()
        if not user:
            session.pop("user", None)
            return jsonify({"error": "AUTH_REQUIRED", "message": "Please log in again."}), 401
        conn.execute(
            """INSERT INTO student_profiles (user_id, skills, cgpa, interests, projects, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                 skills=excluded.skills, cgpa=excluded.cgpa, interests=excluded.interests,
                 projects=excluded.projects, updated_at=CURRENT_TIMESTAMP""",
            (user["id"], skills, cgpa, interests, projects)
        )
        conn.execute(
            """INSERT INTO predictions
               (user_id, skills, cgpa, interests, projects, result_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user["id"], skills, cgpa, interests, projects, json.dumps(result))
        )

    return jsonify(result)


@app.route("/api/my-profile")
def my_profile():
    user_email = session.get("user")
    if not user_email:
        return jsonify({"error":"AUTH_REQUIRED"}), 401
    with get_db_connection() as conn:
        row = conn.execute("""SELECT p.skills, p.cgpa, p.interests, p.projects, p.updated_at
                              FROM student_profiles p JOIN users u ON u.id=p.user_id
                              WHERE u.email=?""", (user_email,)).fetchone()
    return jsonify(dict(row) if row else {})

@app.route("/api/my-predictions")
def my_predictions():
    user_email = session.get("user")
    if not user_email:
        return jsonify({"error": "AUTH_REQUIRED"}), 401
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT p.id, p.cgpa, p.created_at, p.result_json
               FROM predictions p JOIN users u ON u.id = p.user_id
               WHERE u.email = ? ORDER BY p.id DESC""",
            (user_email,)
        ).fetchall()
    return jsonify([{
        "id": row["id"], "cgpa": row["cgpa"], "created_at": row["created_at"],
        "result": json.loads(row["result_json"])
    } for row in rows])


@app.route("/api/my-predictions", methods=["DELETE"])
def clear_my_predictions():
    """Delete only the signed-in user's prediction history; retain account and profile."""
    user_email = session.get("user")
    if not user_email:
        return jsonify({"error": "AUTH_REQUIRED", "message": "Please log in first."}), 401

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE email = ?", (user_email,)
        ).fetchone()
        if not user:
            session.pop("user", None)
            return jsonify({"error": "AUTH_REQUIRED", "message": "Please log in again."}), 401
        cursor = conn.execute(
            "DELETE FROM predictions WHERE user_id = ?", (user["id"],)
        )
        deleted_count = cursor.rowcount

    return jsonify({"success": True, "deleted_count": deleted_count,
                    "message": "Prediction history cleared. Your account and profile are safe."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
