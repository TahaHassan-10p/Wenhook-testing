/**
 * Script to create test users in Azure AD with emails matching your roles/groups.
 * Requires: Directory.ReadWrite.All or User.ReadWrite.All permission for your app registration.
 * Usage: node createTestUsers.js
 */

const axios = require("axios");
require("dotenv").config({
  path: "C:/Users/abiha.anjum/Desktop/ATS - Project 1/ats/backend/.env",
}); // Load env vars from absolute path

// Debug: print all loaded env keys/values (except secrets)
console.log("All loaded env keys:");
Object.keys(process.env).forEach((key) => {
  if (key.toLowerCase().includes("secret")) {
    console.log(`${key}: [HIDDEN]`);
  } else {
    console.log(`${key}: ${process.env[key]}`);
  }
});

// Azure AD app registration credentials (set these as env vars or hardcode for testing)
const TENANT_ID = process.env.TENANT_ID;
const CLIENT_ID = process.env.CLIENT_ID;

// Fix: Try both CLIENT_SECRET and SECRET for backwards compatibility
const CLIENT_SECRET = process.env.CLIENT_SECRET || process.env.SECRET;

console.log("TENANT_ID:", TENANT_ID);
console.log("CLIENT_ID:", CLIENT_ID);
console.log("CLIENT_SECRET:", CLIENT_SECRET ? "[HIDDEN]" : undefined);

// Define your actual groups here
const roleNames = [
  "super admin",
  "recruiters",
  "interviewers",
  "jd approvers",
  "directors",
  "requisitors",
  "HR admin",
];

// Helper to get an access token for Microsoft Graph
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

// Create a user in Azure AD
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
    for (const role of roleNames) {
      // Remove spaces and make lowercase for username
      const username = role.replace(/\s+/g, "").toLowerCase();
      await createUser(token, username, role);
    }
    console.log("Done.");
  } catch (err) {
    // Enhanced error logging for top-level errors
    if (err.response && err.response.data) {
      console.error(
        "Top-level error response:",
        JSON.stringify(err.response.data, null, 2)
      );
    }
    console.error("Failed to create test users:", err.message, err.stack);
  }
})();
