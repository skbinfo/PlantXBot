<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>About | Your Platform Name</title>
    <!-- Replace with your actual CSS path -->
    <!-- <link rel="stylesheet" href="css/style.css" /> -->
    <link rel="icon" href="images/favicon.png" type="image/png">
   <style>
        /* ===== GLOBAL & TYPOGRAPHY ===== */
        :root {
            /* Branding Colors - Update these to match your theme */
            --brand-blue: #2c3e50;
            --mid-blue:   #34495e;
            --light-blue: #3498db;
            
            --text-dark:  #333;
            --text-light: #fff;
            --gray:       #f7f8fa;
            --border-color: #e9ecef;
            --transition: 0.3s ease-in-out;
            --radius: 0.75rem;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --max-w: 1200px;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.7;
            color: var(--text-dark);
            background-color: var(--gray);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: var(--max-w);
            margin: 0 auto;
            padding: 0 1rem;
        }

        h1, h2, h3 {
            color: var(--brand-blue);
            line-height: 1.2;
            font-weight: 700;
        }

        h1 { font-size: 2.8rem; text-align: center; margin-bottom: 1rem; }
        h2 { font-size: 2.2rem; text-align: center; margin-bottom: 2.5rem; border-bottom: 2px solid var(--border-color); padding-bottom: 1rem; }
        h3 { font-size: 1.5rem; color: var(--mid-blue); margin-bottom: 1rem; }

        p { margin-bottom: 1rem; max-width: 80ch; }
        a { color: var(--mid-blue); text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* ===== SECTION STYLING ===== */
        section {
            padding: 4rem 0;
        }
        .section-intro {
            text-align: center;
            padding-bottom: 3rem;
        }
        .lead {
            font-size: 1.2rem;
            color: black;
            max-width: 900px;
            margin: 0 auto;
        }

        /* ===== GRID & CARD STYLES ===== */
        .grid {
            display: grid;
            gap: 2rem;
        }
        .grid-cols-2 { grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }
        .grid-cols-3 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }

        .card {
            background: var(--text-light);
            padding: 2rem;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
        }
        .card img {
            display: block;
            max-width: 100%;
            border-radius: var(--radius);
            margin-top: 1.5rem;
            box-shadow: var(--shadow);
        }

        /* ===== DATABASE TABLE ===== */
        .db-table-wrapper {
            overflow-x: auto;
            box-shadow: var(--shadow);
            border-radius: var(--radius);
            border: 1px solid var(--border-color);
        }
        .db-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--text-light);
        }
        .db-table th, .db-table td {
            padding: 1rem 1.25rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        .db-table th {
            background-color: var(--brand-blue);
            color: var(--text-light);
            font-weight: 600;
        }
        .db-table tbody tr:nth-child(even) {
            background-color: var(--gray);
        }
        .db-table tbody tr:hover {
            background-color: #e6f0fa;
        }

        /* ===== FOOTER ===== */
        footer {
            background: var(--brand-blue);
            color: var(--text-light);
            padding: 2rem 1rem;
            margin-top: 2rem;
            text-align: center;
        }
    </style>
    
</head>
<body>

<!-- Include Header -->
<?php include 'header.html'; ?>

<main>
    <section style="background: var(--text-light); padding-top: 3rem;">
        <div class="container section-intro">
            <h1>About Our Platform</h1>
            <p class="lead">
                The Data Intelligence Hub is an AI-powered suite of tools designed to simplify access to specialized enterprise data. Using natural language processing, users can explore complex datasets, visualize trends, and retrieve specific records across numerous domains without writing a single line of SQL code.
            </p>
        </div>
    </section>
    
    <section>
        <div class="container">
            <h2>Technological Framework</h2>
            <div class="grid grid-cols-3">
                <div class="card">
                    <h3>Large Language Models (LLMs)</h3>
                    <p>Our system uses a dual-LLM strategy for optimal speed and accuracy. A lightweight model classifies user intent in real-time, while a more powerful reasoning model handles complex logic, code generation, and data summarization tasks.</p>
                </div>
                <div class="card">
                    <h3>Dynamic Prompt Engineering</h3>
                    <p>We utilize an advanced templating framework. Prompts are dynamically injected with database schemas, domain-specific knowledge graphs, and conversation history. This ensures that the AI's outputs are always contextually aware and syntactically valid.</p>
                </div>
                <div class="card">
                    <h3>Retrieval-Augmented Generation (RAG)</h3>
                    <p>To answer questions beyond structured data, we employ a RAG architecture. User queries are matched against a vector database containing embedded documentation and articles, providing the system with relevant context to formulate accurate, fact-based answers.</p>
                </div>
            </div>
        </div>
    </section>

    <section style="background: var(--text-light);">
        <div class="container">
            <h2>How It Works: The Workflow</h2>
            <p class="lead" style="margin-bottom: 3rem;">Every query travels through an orchestrated pipeline to ensure accuracy and relevance, transforming raw questions into clear, actionable insights.</p>
            
            <div class="grid grid-cols-2">
                <div class="card">
                    <h3>1. Intent Classification</h3>
                    <p>The system first determines the nature of your request (e.g., retrieving specific data, asking a general question, or simple conversation). This routing ensures the most efficient processing path is selected.</p>
                </div>
                <div class="card">
                    <h3>2. Query Planning & Generation</h3>
                    <p>For data-related requests, the AI analyzes the database schema and creates a precise execution plan. It then translates your natural language into optimized SQL queries targeting the correct tables and columns.</p>
                </div>
                <div class="card">
                    <h3>3. Secure Execution & Processing</h3>
                    <p>Generated queries are executed securely on read-only database replicas. The raw results are processed using high-performance data libraries to compute statistics, filter outliers, and prepare the data for presentation.</p>
                </div>
                <div class="card">
                    <h3>4. Summarization & Delivery</h3>
                    <p>Finally, the system transforms the raw data and statistical reports into a human-readable summary. The response is delivered instantly, often accompanied by downloadable files for deeper analysis.</p>
                </div>
            </div>
        </div>
    </section>

    <section>
        <div class="container">
            <h2>Integrated Data Sources</h2>
            <p class="lead" style="margin-bottom: 3rem;">Our platform connects to a wide array of specialized databases, allowing for comprehensive cross-domain analysis.</p>
            <div class="db-table-wrapper">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Dataset Name</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Customer Metrics DB</td><td>Contains historical customer interaction logs and demographic data.</td></tr>
                        <tr><td>Inventory Master</td><td>Real-time tracking of product stock levels across multiple warehouses.</td></tr>
                        <tr><td>Financial Records</td><td>Quarterly revenue reports, expense tracking, and fiscal projections.</td></tr>
                        <tr><td>Logistics Data</td><td>Shipping routes, delivery times, and carrier performance metrics.</td></tr>
                        <tr><td>Employee Directory</td><td>Organizational hierarchy, department listings, and contact info.</td></tr>
                        <tr><td>Support Tickets</td><td>Archive of customer support requests, resolutions, and response times.</td></tr>
                        <tr><td>Sales Pipeline</td><td>Active leads, conversion rates, and deal stages.</td></tr>
                        <tr><td>Web Analytics</td><td>Visitor traffic, bounce rates, and session duration data.</td></tr>
                        <tr><td>Market Research</td><td>Competitor analysis and industry trend reports.</td></tr>
                        <tr><td>Legacy Archives</td><td>Digitized records from pre-2020 operations.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </section>

    <section style="background: var(--text-light);">
        <div class="container">
            <h2>System Architecture</h2>
            <div class="grid grid-cols-2" style="align-items: center;">
                <div>
                    <h3>Modular Design</h3>
                    <p>To provide flexibility, our system is built on a modular architecture:</p>
                    <ul>
                        <li><strong>Specialized Modules:</strong> Dedicated components handle specific data types (e.g., financial vs. logistical) to ensure high precision.</li>
                        <li><strong>Integrative Layer:</strong> A central orchestration layer allows users to query across different modules seamlessly, enabling complex comparative analysis.</li>
                    </ul>
                </div>
                <div class="card" style="text-align: center; padding: 1rem;">
                    <!-- Placeholder for Architecture Diagram -->
                    <div style="background: #eee; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: var(--radius);">
                        <span style="color: #777;">[Architecture Diagram Placeholder]</span>
                    </div>
                    <p style="margin-top: 1rem;"><strong>Figure:</strong> The system uses a hub-and-spoke model to connect disparate data sources.</p>
                </div>
            </div>
        </div>
    </section>
    
    <section>
        <div class="container">
            <h2>Infrastructure</h2>
            <div class="card">
                <p>The entire backend framework is deployed on enterprise-grade servers running <strong>Linux</strong>. The system relies on scalable cloud infrastructure with auto-scaling capabilities, ensuring it can handle high concurrency and large data volumes. Robust caching mechanisms and optimized database indexing guarantee sub-second response times for most queries.</p>
            </div>
        </div>
    </section>
</main>

<!-- Include Footer -->
<?php include 'footer.html';?>

</body>
</html>
