// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "Projects",
          description: "A growing collection of my cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "Repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "Curriculum Vitae of Jha Saket Sunil",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "dropdown-kalppo",
              title: "Kalppo",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/experience/kalppo/";
              },
            },{id: "post-semantic-search-with-faiss-from-theory-to-production",
        
          title: "Semantic Search with FAISS From Theory to Production",
        
        description: "Semantic search has revolutionized how we find and retrieve information. Unlike traditional keyword-based search, semantic search understands the meaning and context behind queries, enabling more intelligent and relevant results. In this comprehensive guide, we&#39;ll explore how to implement production ready semantic search systems using FAISS (Facebook AI Similarity Search) and sentence transformers.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/semantic-search-faiss/";
          
        },
      },{id: "post-building-scalable-backend-systems-with-fastapi-and-postgresql",
        
          title: "Building Scalable Backend Systems with FastAPI and PostgreSQL",
        
        description: "Building scalable backend systems is one of the most critical aspects of modern software development. In this comprehensive guide, we&#39;ll explore how to design and implement robust backend architectures using FastAPI and PostgreSQL that can handle millions of requests while maintaining high performance and reliability.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/fastapi-postgresql/";
          
        },
      },{id: "post-winning-kaggle-competition-is-a-piece-of-cake",
        
          title: "Winning Kaggle Competition is a Piece of Cake!",
        
        description: "This post is about how I won the KaggleX Skill Assessment Machine Learning Competition. Disclaimer I was just a beginner in machine learning and Kaggle competitions when I started. This competition ran for about 20 days, and I improved day by day and climbed up the leaderboard. Initially, I was ranked around 368-ish on the private leaderboard, but when the competition ended, the tables turned and I was Rank 1.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/kaggle-x-skill-assesment/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-winner-kaggle-x-skill-assessment-machine-learning-competition-rank-1-1846",
          title: 'Winner, Kaggle x Skill Assessment Machine Learning Competition – Rank 1/1846',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/kaggle-skill-assessment-ml-competition/";
            },},{id: "news-completed-my-ai-engineer-internship-at-kalppo",
          title: 'Completed My AI Engineer Internship at Kalppo!',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/kalppo-internship/";
            },},{id: "projects-paragon-parallel-graph-engine",
          title: 'PARAGON – Parallel Graph Engine',
          description: "A high-performance C++ parallel graph processing engine for large-scale graph analytics",
          section: "Projects",handler: () => {
              window.location.href = "/projects/paragon/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73%61%6B%65%74%6A%68%61%30%33%32%34@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-whatsapp',
        title: 'whatsapp',
        section: 'Socials',
        handler: () => {
          window.open("https://wa.me/+91-9370456334", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/saketjha34", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/saketjha34", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
