<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Team | Your Brand Name</title>
    <!-- Replace href with your favicon path -->
    <link rel="icon" href="images/favicon.png" type="image/png"> 
    <!-- If you have an external CSS file, uncomment the line below -->
    <!-- <link rel="stylesheet" href="css/style.css" /> -->
   <style>
        /* ===== GLOBAL & TYPOGRAPHY ===== */
        :root {
            /* Customize your brand colors here */
            --brand-blue: #333333; /* Primary Color */
            --mid-blue:   #555555; /* Secondary Color */
            --light-blue: #777777; /* Accent Color */
            
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
            font-size: 0.9rem;
            color: #555;
            margin-top: 0.5rem;
        }

        /* ===== FOOTER STYLING (If not included via PHP) ===== */
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
<!-- Ensure 'header.html' exists in your directory -->
<?php include 'header.html'; ?>

<section style="background: var(--text-light); padding-top: 3rem;">
    <div class="container section-intro">
        <h1>Meet Our Team</h1>  
        <p class="lead">
            Insert your team's mission statement here. For example: "Our multidisciplinary team combines expertise in [Field A], [Field B], and [Field C]. Together, we develop innovative tools to advance research and unlock new insights."
        </p>
    </div> 

    <div class="container">
        <div class="grid grid-cols-4">
            <!-- Team Member 1 -->
            <div class="card">
                <!-- Using placeholder image service -->
                <img src="https://via.placeholder.com/150" alt="Team Member Photo">
                <h3>John Doe</h3>
                <p>Lead Developer</p>
                <p>Organization Name</p>
            </div>
            
            <!-- Team Member 2 -->
            <div class="card">
                <img src="https://via.placeholder.com/150" alt="Team Member Photo">
                <h3>Jane Smith</h3>
                <p>Project Manager</p>
                <p>Organization Name</p>
            </div>
            
            <!-- Team Member 3 -->
            <div class="card">
                <img src="https://via.placeholder.com/150" alt="Team Member Photo">
                <h3>Dr. Alex Johnson</h3>
                <p>Senior Scientist</p>
                <p>Organization Name</p>
            </div>
            
            <!-- Team Member 4 -->
            <div class="card">
                <img src="https://via.placeholder.com/150" alt="Team Member Photo">
                <h3>Sarah Lee</h3>
                <p>Research Scholar</p>
                <p>Organization Name</p>
            </div>
        </div>
    </div>       
</section>

<!-- Include Footer -->
<!-- Ensure 'footer.html' exists in your directory -->
<?php include 'footer.html';?>

</body>
</html>
