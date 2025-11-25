<DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Team | Plant Regulatory Elements</title>
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
        .grid-cols-4 { 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
        }

        .card {
            background: var(--text-light);
            padding: 2rem;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            text-align: center;
            min-height: 400px; /* taller card */
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .card img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            margin: 0 auto 1rem auto;
            box-shadow: var(--shadow);
        }

        .card h3 {
            text-align: center;
            margin: 1rem 0 0.5rem 0;
        }

        .card p {
            font-size: 0.9rem; /* smaller font for description */
            color: #555;
            margin-top: 0.5rem;
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

<section style="background: var(--text-light); padding-top: 3rem;">
    <div class="container section-intro">
        <h1>Meet Our Team</h1>  
     <p class="lead">
    Our multidisciplinary team combines expertise in plant genomics, computational biology, and AI-driven data science. Together, we develop innovative tools and databases to advance plant molecular research. We are committed to fostering collaboration and enabling the scientific community to unlock new insights in plant biology.
</p>
    </div> 

    <div class="container">
        <div class="grid grid-cols-4">
            <div class="card">
                <img src="images/team/kanka.jpg" alt="tRF Bot">
                <h3>Kanka Mukherjee</h3>
                <p>Project Associate-I</p>
                <p>National Institute of Plant Genome Research</p>
            </div>
            <div class="card">
                <img src="images/team/niyati.jpg" alt="Fusion Bot">
                <h3>Niyati Bisht</h3>
                <p>Project Associate-II</p>
                <p>National Institute of Plant Genome Research</p>

            </div>
            <div class="card">
                <img src="images/team/vivek.png" alt="AMP Bot">
                <h3>Dr. AT Vivek</h3>
                <p>Project Scientist-I</p>
                <p>National Institute of Plant Genome Research</p>

            </div>
            <div class="card">
                <img src="images/team/fiza.jpg" alt="Cotton Bot">
                <h3>Fiza Hamid</h3>
                <p>Ph.D. Scholar</p>
                <p>National Institute of Plant Genome Research</p>
            </div>
        </div>
    </div>       
</section>

<!-- Include Footer -->
<?php include 'footer.html';?>

</body>
</html>


