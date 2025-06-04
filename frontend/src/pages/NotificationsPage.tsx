import React, { useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
} from '@mui/material'
import { Notifications, Warning, Info, CheckCircle, MarkEmailRead } from '@mui/icons-material'
import { useTranslation } from 'react-i18next'
import { useAppDispatch, useAppSelector } from '../hooks/redux'
import { fetchNotifications, markAsRead } from '../store/slices/notificationSlice'
import { useFormatters } from '../utils/formatters'

const NotificationsPage: React.FC = () => {
  const { t } = useTranslation()
  const formatters = useFormatters()
  const dispatch = useAppDispatch()
  const { notifications, isLoading, error } = useAppSelector(state => state.notification)

  useEffect(() => {
    dispatch(fetchNotifications({}))
  }, [dispatch])

  const handleMarkAsRead = (notificationId: number): void => {
    dispatch(markAsRead(notificationId))
  }

  const getPriorityColor = (
    priority: string
  ): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (priority) {
      case 'critical':
        return 'error'
      case 'high':
        return 'warning'
      case 'medium':
        return 'info'
      case 'low':
        return 'success'
      default:
        return 'default'
    }
  }

  const getNotificationIcon = (type: string): React.ReactNode => {
    switch (type) {
      case 'validation_assignment':
        return <CheckCircle />
      case 'urgent_alert':
        return <Warning />
      case 'completion_notification':
        return <Info />
      default:
        return <Notifications />
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        {t('notifications.title')}
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {t('notifications.recentNotifications')}
          </Typography>
          <List>
            {notifications.map(notification => (
              <ListItem
                key={notification.id}
                sx={{
                  border: '1px solid #eee',
                  borderRadius: 1,
                  mb: 1,
                  backgroundColor: notification.isRead ? 'transparent' : '#f5f5f5',
                }}
                secondaryAction={
                  !notification.isRead && (
                    <IconButton
                      edge="end"
                      aria-label={t('notifications.markAsRead')}
                      onClick={() => handleMarkAsRead(notification.id)}
                    >
                      <MarkEmailRead />
                    </IconButton>
                  )
                }
              >
                <ListItemIcon>{getNotificationIcon(notification.notificationType)}</ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1">{notification.title}</Typography>
                      <Chip
                        label={notification.priority}
                        color={getPriorityColor(notification.priority)}
                        size="small"
                      />
                      {!notification.isRead && <Chip label={t('notifications.new')} color="primary" size="small" />}
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {notification.message}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatters.formatDateTime(new Date(notification.createdAt))}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
            {notifications.length === 0 && !isLoading && (
              <ListItem>
                <ListItemText primary={t('notifications.noNotifications')} secondary={t('notifications.allCaughtUp')} />
              </ListItem>
            )}
          </List>
        </CardContent>
      </Card>
    </Box>
  )
}

export default NotificationsPage
