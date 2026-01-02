/**
 * Script to create two recruiter users (recruiter1 and recruiter2) in Azure AD.
 * Usage: node createnewuser.js
 */
console.log("Starting recruiter user creation script...");

const axios = require("axios");
require("dotenv").config({
  path: "C:/Users/abiha.anjum/Desktop/ATS - Project 1/ats/backend/.env",
});

const TENANT_ID = process.env.TENANT_ID;
const CLIENT_ID = process.env.CLIENT_ID;
const CLIENT_SECRET = process.env.CLIENT_SECRET || process.env.SECRET;

async function getGraphToken() {
  const url = `https://login.microsoftonline.com/${TENANT_ID}/oauth2/v2.0/token`;
  const params = new URLSearchParams();
  params.append("client_id", CLIENT_ID);
  params.append("scope", "https://graph.microsoft.com/.default");
  params.append("client_secret", CLIENT_SECRET);
  params.append("grant_type", "client_credentials");

  const res = await axios.post(url, params, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  return res.data.access_token;
}

async function createUser(token, username, displayName) {
  const user = {
    accountEnabled: true,
    displayName: displayName,
    mailNickname: username,
    userPrincipalName: `${username}@10pdev.us`,
    passwordProfile: {
      forceChangePasswordNextSignIn: false,
      password: "TestUser123!",
    },
  };
  try {
    console.log(
      "Attempting to create user with payload:",
      JSON.stringify(user, null, 2)
    );
    const res = await axios.post(
      "https://graph.microsoft.com/v1.0/users",
      user,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      }
    );
    console.log(`Created user: ${user.userPrincipalName}`);
  } catch (err) {
    console.error("Request payload:", JSON.stringify(user, null, 2));
    if (err.response) {
      console.error("Status:", err.response.status);
      console.error("Headers:", JSON.stringify(err.response.headers, null, 2));
      console.error(
        `Error creating user ${user.userPrincipalName}:`,
        JSON.stringify(err.response.data, null, 2)
      );
    } else {
      console.error(
        `Error creating user ${user.userPrincipalName}:`,
        err.message
      );
    }
  }
}

(async () => {
  try {
    const token = await getGraphToken();
    await createUser(token, "recruiter1", "recruiter1");
    await createUser(token, "recruiter2", "recruiter2");
    console.log("Done.");
  } catch (err) {
    if (err.response && err.response.data) {
      console.error(
        "Top-level error response:",
        JSON.stringify(err.response.data, null, 2)
      );
    }
    console.error("Failed to create recruiter users:", err.message, err.stack);
  }
})();
