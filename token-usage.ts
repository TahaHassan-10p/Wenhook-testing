/**
 * Token Usage Service
 * 
 * Provides API client methods for managing user token quotas and usage tracking.
 * Handles:
 * - User token quota retrieval
 * - Admin quota management (update quotas, add purchased tokens)
 * - Token usage history and statistics
 * - Platform-wide token analytics
 */

import { makeAuthenticatedRequest } from './auth';

// API Base URL
const API_BASE_URL = import.meta.env.VITE_API_BASE;

// Types matching backend schemas
export interface UserTokenStats {
  monthly_quota: number;
  tokens_used_this_month: number;
  remaining_monthly: number;
  purchased_tokens_balance: number;
  total_available: number;
  quota_reset_date: string;
  total_tokens_lifetime: number;
}

export interface MonthlyUsageStats {
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost: number;
}

export interface OperationBreakdown {
  operation_type: string;
  tokens: number;
  count: number;
}

export interface DetailedUserTokenStats extends UserTokenStats {
  monthly_stats: MonthlyUsageStats;
  operation_breakdown: OperationBreakdown[];
}

export interface TokenOperationRecord {
  id: number;
  user_id: number;
  project_id?: number;
  operation_type: string;
  ai_provider: string;
  ai_model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost: number;
  created_at: string;
}

export interface TokenOperationHistory {
  operations: TokenOperationRecord[];
  total: number;
  skip: number;
  limit: number;
}

export interface QuotaCheckResponse {
  allowed: boolean;
  remaining_monthly_tokens: number;
  purchased_tokens_balance: number;
  total_available_tokens: number;
  estimated_tokens_required?: number;
  message?: string;
}

export interface PlatformStats {
  total_users: number;
  total_tokens_consumed_this_month: number;
  total_purchased_tokens_balance: number;
  average_usage_per_user: number;
  total_operations_this_month: number;
}

export interface TokenQuotaUpdate {
  monthly_token_quota: number;
}

export interface PurchasedTokensAdd {
  amount: number;
  reason?: string;
}

/**
 * Get current user's token usage statistics
 */
export const getMyTokenUsage = async (): Promise<DetailedUserTokenStats> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/me`,
    { method: 'GET' }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to fetch token usage');
  }
  
  return await response.json();
};

/**
 * Get token usage statistics for a specific user (admin only)
 * @param userId - The ID of the user to retrieve stats for
 */
export const getUserTokenUsage = async (userId: number): Promise<DetailedUserTokenStats> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/users/${userId}`,
    { method: 'GET' }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to fetch user token usage');
  }
  
  return await response.json();
};

/**
 * Update a user's monthly token quota (admin only)
 * @param userId - The ID of the user to update
 * @param newQuota - The new monthly token quota
 */
export const updateUserQuota = async (
  userId: number,
  newQuota: number
): Promise<DetailedUserTokenStats> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/users/${userId}/quota`,
    {
      method: 'PUT',
      body: JSON.stringify({ monthly_token_quota: newQuota }),
      headers: { 'Content-Type': 'application/json' }
    }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to update quota');
  }
  
  return await response.json();
};

/**
 * Add purchased tokens to a user's account (admin only)
 * @param userId - The ID of the user to add tokens to
 * @param amount - The number of tokens to add
 * @param reason - Optional reason for the token addition
 */
export const addPurchasedTokens = async (
  userId: number,
  amount: number,
  reason?: string
): Promise<DetailedUserTokenStats> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/users/${userId}/purchased-tokens`,
    {
      method: 'POST',
      body: JSON.stringify({ amount, reason }),
      headers: { 'Content-Type': 'application/json' }
    }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to add purchased tokens');
  }
  
  return await response.json();
};

/**
 * Get token operation history for a user (admin only)
 * @param userId - The ID of the user to retrieve operations for
 * @param skip - Number of records to skip (pagination)
 * @param limit - Maximum number of records to return
 */
export const getUserOperations = async (
  userId: number,
  skip: number = 0,
  limit: number = 50
): Promise<TokenOperationHistory> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/users/${userId}/operations?skip=${skip}&limit=${limit}`,
    { method: 'GET' }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to fetch operations');
  }
  
  return await response.json();
};

/**
 * Get platform-wide token usage statistics (admin only)
 */
export const getPlatformStats = async (): Promise<PlatformStats> => {
  const response = await makeAuthenticatedRequest(
    `${API_BASE_URL}/api/v1/token-quota/platform-stats`,
    { method: 'GET' }
  );
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to fetch platform stats');
  }
  
  return await response.json();
};

/**
 * Format token count to human-readable format
 * @param tokens - Raw token count
 * @returns Formatted string (e.g., "1.5M", "500K", "1,234")
 */
export const formatTokens = (tokens: number | undefined | null): string => {
  if (tokens === undefined || tokens === null) {
    return '0';
  }
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1)}M`;
  } else if (tokens >= 1_000) {
    return `${(tokens / 1_000).toFixed(1)}K`;
  }
  return tokens.toLocaleString();
};

/**
 * Calculate percentage of quota used
 * @param used - Tokens used
 * @param total - Total quota
 * @returns Percentage (0-100)
 */
export const calculateUsagePercentage = (used: number, total: number): number => {
  if (total === 0) return 0;
  return Math.round((used / total) * 100);
};

/**
 * Format date for quota reset display
 * @param dateString - ISO date string
 * @returns Formatted date string
 */
export const formatResetDate = (dateString: string | undefined | null): string => {
  if (!dateString) {
    return '-';
  }
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    return '-';
  }
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
};
