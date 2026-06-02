/**
 * Discord Activity OAuth2 Token Exchange
 * 
 * This serverless function exchanges the authorization code
 * from Discord's Embedded App SDK for an access token.
 * 
 * Required environment variables (set in Vercel):
 * - DISCORD_CLIENT_ID
 * - DISCORD_CLIENT_SECRET
 */

export default async function handler(req, res) {
  // CORS headers for iframe access
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { code } = req.body;

  if (!code) {
    return res.status(400).json({ error: 'Missing authorization code' });
  }

  const clientId = process.env.DISCORD_CLIENT_ID;
  const clientSecret = process.env.DISCORD_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    console.error('Missing DISCORD_CLIENT_ID or DISCORD_CLIENT_SECRET');
    return res.status(500).json({ error: 'Server configuration error' });
  }

  try {
    // Exchange code for access token
    const tokenResponse = await fetch('https://discord.com/api/oauth2/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'authorization_code',
        code: code,
      }),
    });

    if (!tokenResponse.ok) {
      const errorData = await tokenResponse.text();
      console.error('Token exchange failed:', tokenResponse.status, errorData);
      return res.status(tokenResponse.status).json({ 
        error: 'Token exchange failed',
        details: errorData 
      });
    }

    const tokenData = await tokenResponse.json();

    // Fetch user info with the access token
    const userResponse = await fetch('https://discord.com/api/v10/users/@me', {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
      },
    });

    let user = null;
    if (userResponse.ok) {
      user = await userResponse.json();
    }

    return res.status(200).json({
      access_token: tokenData.access_token,
      token_type: tokenData.token_type,
      expires_in: tokenData.expires_in,
      refresh_token: tokenData.refresh_token,
      scope: tokenData.scope,
      user: user,
    });

  } catch (error) {
    console.error('OAuth2 error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
