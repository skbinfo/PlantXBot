<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>About | Plant Regulatory Elements</title>
    <link rel="stylesheet" href="css/style.css" />
    <link rel="icon" href="images/favicon/logo.png" type="image/png">
   <style>
        /* ===== GLOBAL & TYPOGRAPHY ===== */
        :root {
            --brand-blue: #004c97;
            --mid-blue:   #0063c6;
            --light-blue: #1490ff;
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

        /* ===== NAVBAR ===== */
        .navbar {
            background: var(--brand-blue);
            padding: 1rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .navbar .container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand-logo-title { display: flex; align-items: center; }
        .logo-circle {
            width: 50px; height: 50px;
            border-radius: 50%; object-fit: cover;
            margin-right: 15px;
        }
        .brand {
            font-size: 1.5rem; font-weight: bold;
            color: var(--text-light); text-decoration: none;
        }
        nav a {
            margin-left: 20px; text-decoration: none;
            color: var(--text-light); font-size: 1rem; font-weight: 500;
            transition: var(--transition);
        }
        nav a:hover { opacity: 0.8; }

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

        /* ===== SCROLL-TO-TOP BUTTON ===== */
        #scrollTopBtn {
            display: none; position: fixed;
            bottom: 30px; right: 30px;
            z-index: 99; font-size: 1.2rem;
            border: none; outline: none;
            background-color: var(--brand-blue);
            color: white; cursor: pointer;
            width: 50px; height: 50px;
            border-radius: 50%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: background-color 0.3s, transform 0.3s;
        }
        #scrollTopBtn:hover {
            background-color: var(--mid-blue);
            transform: translateY(-3px);
        }
    </style>
    
</head>
<body>

<!-- Include Header -->
<?php include 'header.html'; ?>

<section style="background: var(--text-light); padding-top: 3rem;">
        <div class="container section-intro">
            <h1>About Our Platform</h1>
            <p class="lead">
                The Plant Genomics Chatbot Hub is an AI-powered suite of tools designed to simplify access to specialized plant genomics data. Using natural language, researchers and students can now explore complex datasets related to tRFs, fusion transcripts, and other molecular classes across numerous plant species without writing a single line of code.
            </p>
        </div>
    </section>
    
    <section>
        <div class="container">
            <h2>Technological Framework</h2>
            <div class="grid grid-cols-3">
                <div class="card">
                    <h3>Large Language Models (LLMs)</h3>
                    <p>Our system uses a dual-LLM strategy via the Groq™ inference engine for speed and accuracy. A lightweight model (Llama-3.1-8B-Instant) classifies user intent, while a more powerful model (GPT-OSS-120B or Llama-3.3-70B-Versatile) handles complex reasoning, SQL generation, and summarization.</p>
                </div>
                <div class="card">
                    <h3>Prompt Engineering</h3>
                    <p>We use advanced prompt engineering with LangChain's `PromptTemplate` framework. Prompts are dynamically injected with database schemas, knowledge graphs, and conversation history to guide the LLM, ensuring outputs are contextually aware, syntactically valid, and semantically coherent.</p>
                </div>
                <div class="card">
                    <h3>Retrieval-Augmented Generation (RAG)</h3>
                    <p>To answer questions beyond the scope of our SQL databases, we employ a RAG architecture. User queries are matched against a ChromaDB vector store containing embedded research articles, providing the LLM with relevant context to formulate accurate, fact-based answers.</p>
                </div>
            </div>
        </div>
    </section>

    <section style="background: var(--text-light);">
        <div class="container">
            <h2>How It Works: A Multi-Stage Pipeline</h2>
            <p class="lead" style="margin-bottom: 3rem;">Every query is processed through an orchestrated workflow to ensure accuracy and relevance, transforming your question into a clear, data-backed answer.</p>
            
            <div class="grid grid-cols-2">
                <div class="card">
                    <h3>1. Intent Classification</h3>
                    <p>First, your query is classified to determine its intent (e.g., data retrieval, metadata question, or general conversation). This allows the system to choose the most efficient path—either querying the database, consulting our knowledge base via RAG, or providing a direct conversational reply.</p>
                </div>
                <div class="card">
                    <h3>2. SQL Query Planning</h3>
                    <p>For data-related questions, the powerful LLM, guided by the database schema and a curated knowledge graph, translates your natural language query into one or more precise, executable SQL statements. This plan ensures the correct tables and columns are queried.</p>
                </div>
                <div class="card">
                    <h3>3. Data Retrieval & Processing</h3>
                    <p>The generated SQL queries are safely executed on read-only copies of our SQLite databases. The raw data is then processed using pandas to compute statistical summaries and prepare it for interpretation, with large datasets being intelligently sampled for efficiency.</p>
                </div>
                <div class="card">
                    <h3>4. Summary Generation</h3>
                    <p>Finally, the LLM transforms the structured data and statistical reports into a coherent, easy-to-understand conversational summary. The final response, along with a link to download the full dataset as a CSV, is delivered to you in a clean JSON format.</p>
                </div>
            </div>
        </div>
    </section>

    <section>
        <div class="container">
            <h2>Integrated Databases</h2>
            <p class="lead" style="margin-bottom: 3rem;">Our chatbots are connected to a comprehensive suite of specialized, in-house plant genomics databases developed by our research team.</p>
            <div class="db-table-wrapper">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Database Name</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>PbtRFdb</td><td>Documents biotic stress-responsive tRNA-derived fragments (tRFs).</td></tr>
                        <tr><td>PtRFdb</td><td>Provides tRF sequences from multiple species with expression and functional annotations.</td></tr>
                        <tr><td>PtncRNAdb</td><td>Contains plant tRNA-derived fragments with expression profiles and functional annotations.</td></tr>
                        <tr><td>AtFusionDB</td><td>Catalogs Arabidopsis thaliana fusion transcripts with detailed metadata.</td></tr>
                        <tr><td>PFusionDB</td><td>A repository of fusion transcripts from multiple plant species.</td></tr>
                        <tr><td>PlantPepDB</td><td>Houses plant-derived antimicrobial peptides (AMPs) and their properties.</td></tr>
                        <tr><td>Athisomir</td><td>Contains Arabidopsis isomiR profiles.</td></tr>
                        <tr><td>Cotton ncRNA Atlas</td><td>Contains data on cotton lncRNAs and miRNAs.</td></tr>
                        <tr><td>AlnC</td><td>A collection of long intergenic non-coding RNAs (lincRNAs).</td></tr>
                        <tr><td>ANNinter</td><td>Comprises annotated RNA-RNA interactions.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </section>

    <section style="background: var(--text-light);">
        <div class="container">
            <h2>Chatbot Architecture</h2>
            <div class="grid grid-cols-2" style="align-items: center;">
                <div>
                    <h3>Single vs. Multi-Database Bots</h3>
                    <p>To provide both focused and integrative analyses, our bots are organized into two classes:</p>
                    <ul>
                        <li><strong>Single-Database Bots:</strong> Tightly coupled to the schema of one resource (e.g., the AtFusionDB Bot). This ensures high precision for exploring a single dataset.</li>
                        <li><strong>Multi-Database Bots:</strong> Designed to query across multiple databases with a shared molecular theme (e.g., the tRF Bot). This enables powerful comparative analyses.</li>
                    </ul>
                </div>
                <div class="card" style="text-align: center;">
                    
                    <p><strong>Figure:</strong> Chatbots are categorized as either single-database specialists or multi-database integrators to support different research queries.</p>
                </div>
            </div>
        </div>
    </section>
    
    <section>
        <div class="container">
            <h2>Computational Resources</h2>
            <div class="card">
                <p>The entire backend framework is deployed on a dedicated server running <strong>Ubuntu 22.04 LTS</strong>. The system is equipped with multi-core CPUs and sufficient RAM to handle concurrent user requests, complex data processing with pandas, and efficient in-memory caching. This robust infrastructure ensures a responsive and scalable experience for the scientific community.</p>
                 
            </div>
        </div>
    </section>

</main>

<!-- Include Footer -->
<?php include 'footer.html';?>

</body>
</html>
