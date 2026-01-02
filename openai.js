require('dotenv').config(); // Load .env from backend directory

const OpenAI = require("openai");

// Verify API key
if (!process.env.OPENAI_API_KEY) {
  console.error("ERROR: OPENAI_API_KEY is missing in .env");
  process.exit(1);
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

async function parseAndScoreResume(resumeText, jobTitle, jobDescription) {
  // 🔹 Normalize date ranges (e.g., "2018–2021" → "2018 - 2021")
  resumeText = resumeText.replace(/(\d{4})\s*[-–]\s*(\d{4}|Present)/g, "$1 - $2");

  const prompt = `
    Resume: ${resumeText}
    Job Title: ${jobTitle}
    Job Description: ${jobDescription}
    Parse the resume and extract:
    - All relevant skills (as a comprehensive array, but if there are too many, consolidate or group them into broader categories)
    - All work experiences (as an array of objects with:
        company,
        title (position),
        dates (start and end),
        duration (number of years/months worked),
        description (ONLY the main highlight or achievement for each job, not a full list)
      )
      IMPORTANT:
      - For each work experience, return company, title, and any dates found.
      - For description, only include the single most important highlight/achievement for that job.
      - Do NOT skip experiences just because a date is missing.
      - Return each job as a bullet (array object).
      - Example experience object:
        {
          "company": "Acme Corp",
          "title": "Software Engineer",
          "dates": "Jan 2020 - Mar 2022",
          "duration": "2 years 3 months",
          "description": "Led migration to cloud infrastructure"
        }
    - Score the candidate's fit for the specific job title described above (0-100)

    Respond ONLY with JSON:
    {
      "skills": [ ... ], // consolidated if too many
      "experience": [
        {
          "company": "...",
          "title": "...",
          "dates": "...",
          "duration": "...",
          "description": "main highlight only"
        }, ...
      ],
      "score": <number>
    }
  `;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" }
    });

    console.log("===== OpenAI RAW RESPONSE =====");
    console.log(JSON.stringify(response, null, 2));
    console.log("===== END RAW RESPONSE =====");

    const content = response.choices?.[0]?.message?.content;
    if (!content) {
      throw new Error("No response from OpenAI");
    }

    // Defensive JSON parsing
    let parsed;
    try {
      parsed = JSON.parse(content);
    } catch (err) {
      const fixed = content.match(/\{[\s\S]*\}/); // extract JSON block only
      parsed = fixed ? JSON.parse(fixed[0]) : {};
    }

    return parsed;

  } catch (err) {
    console.error("OpenAI API Error:", err.message);
    return { error: "Failed to parse AI response", details: err.message };
  }
}

module.exports = { parseAndScoreResume };

